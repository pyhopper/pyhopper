PyTorch (ResNet34 with CIFAR-10) 
-------------------------------------

A typical use-case with PyTorch would look something like this

.. code-block:: python

    import numpy as np
    import torchvision
    import torch
    import torch.utils.data
    from torch.utils.data import SubsetRandomSampler
    from tqdm.auto import tqdm

    import pyhopper


    def get_cifar_loader(batch_size, erasing_prob, for_validation):
        mean = np.array([125.30691805, 122.95039414, 113.86538318]) / 255.0
        std = np.array([62.99321928, 62.08870764, 66.70489964]) / 255.0

        train_transform = torchvision.transforms.Compose(
            [
                torchvision.transforms.RandomCrop(32, padding=4),
                torchvision.transforms.RandomHorizontalFlip(),
                torchvision.transforms.ToTensor(),
                torchvision.transforms.RandomErasing(p=erasing_prob),
                torchvision.transforms.Normalize(mean, std),
            ]
        )
        test_transform = torchvision.transforms.Compose(
            [
                torchvision.transforms.ToTensor(),
                torchvision.transforms.Normalize(mean, std),
            ]
        )

        dataset_dir = "~/.torchvision/datasets/CIFAR10"
        train_dataset = torchvision.datasets.CIFAR10(
            dataset_dir, train=True, transform=train_transform, download=True
        )
        train_sampler, test_sampler = torch.utils.data.RandomSampler(train_dataset), None
        if for_validation:
            test_dataset = torchvision.datasets.CIFAR10(
                dataset_dir, train=True, transform=test_transform, download=True
            )
            indices = np.random.default_rng(12345).permutation(len(train_dataset))
            valid_size = int(0.05 * len(train_dataset))
            train_idx, valid_idx = indices[valid_size:], indices[:valid_size]
            train_sampler = SubsetRandomSampler(train_idx)
            test_sampler = SubsetRandomSampler(valid_idx)
        else:
            test_dataset = torchvision.datasets.CIFAR10(
                dataset_dir, train=False, transform=test_transform, download=True
            )

        train_loader = torch.utils.data.DataLoader(
            train_dataset,
            batch_size=batch_size,
            sampler=train_sampler,
            num_workers=4,
            pin_memory=True,
            drop_last=True,
        )
        test_loader = torch.utils.data.DataLoader(
            test_dataset,
            batch_size=256,
            sampler=test_sampler,
            num_workers=4,
            pin_memory=True,
            drop_last=False,
        )
        return train_loader, test_loader


    def training_epoch(model, optimizer, scheduler, loss_fn, train_loader, for_validation):
        model.train()

        correct_samples = 0
        num_samples = 0
        total_loss = 0
        prog_bar = tqdm(total=len(train_loader), disable=not for_validation)
        for step, (data, targets) in enumerate(train_loader):
            data = data.cuda()
            targets = targets.cuda()

            optimizer.zero_grad()
            outputs = model(data)
            loss = loss_fn(outputs, targets)
            total_loss += loss.item()
            with torch.no_grad():
                _, preds = torch.max(outputs, dim=1)
                correct_samples += preds.eq(targets).sum().item()
                num_samples += data.size(0)
            loss.backward()
            optimizer.step()
            scheduler.step()

            prog_bar.update(1)
            prog_bar.set_description_str(
                f"loss={total_loss/(step+1):0.3f}, train_acc={100*correct_samples/num_samples:0.2f}%"
            )
        prog_bar.close()


    def evaluate(model, data_loader):
        model.eval()

        with torch.no_grad():
            num_samples = 0
            correct_samples = 0

            for step, (data, targets) in enumerate(data_loader):
                data = data.cuda()
                targets = targets.cuda()

                outputs = model(data)
                _, preds = torch.max(outputs, dim=1)
                correct_samples += preds.eq(targets).sum().item()
                num_samples += data.size(0)

        return float(correct_samples / num_samples)


    def train_cifar10(params, for_validation=True):
        model = torchvision.models.resnet34(pretrained=False, num_classes=10)
        model.conv1 = torch.nn.Conv2d(
            3, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False
        )
        model.maxpool = torch.nn.Identity()
        model = model.cuda()

        loss_fn = torch.nn.CrossEntropyLoss()

        # optimizer
        optimizer = torch.optim.SGD(
            model.parameters(),
            lr=params["lr"],
            momentum=0.9,
            weight_decay=params["weight_decay"],
            nesterov=True,
        )

        train_loader, val_loader = get_cifar_loader(
            128, params["erasing_prob"], for_validation
        )
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, len(train_loader) * 100, eta_min=params["eta_min"]
        )

        for e in range(100):
            training_epoch(
                model, optimizer, scheduler, loss_fn, train_loader, for_validation
            )
            if not for_validation:
                val_acc = evaluate(model, val_loader)
                print(f"epoch {e} val_acc={100*val_acc:0.2f}%")

        return evaluate(model, val_loader)


    if __name__ == "__main__":
        search = pyhopper.Search(
            {
                "lr": pyhopper.float(0.5, 0.05, precision=1, log=True),
                "eta_min": pyhopper.choice([0, 1e-4, 1e-3, 1e-2], is_ordinal=True),
                "weight_decay": pyhopper.float(1e-6, 1e-2, log=True, precision=1),
                "erasing_prob": pyhopper.float(0, 1, precision=1),
            }
        )
        best_params = search.run(
            train_cifar10,
            direction="max",
            runtime="24h",
            n_jobs="per-gpu",
        )
        test_acc = train_cifar10(best_params, for_validation=False)
        print(f"Tuned params: Test accuracy = {100 * test_acc}")

.. note::

    The original `ResNet paper <https://arxiv.org/pdf/1512.03385.pdf>`_ reported an accuracy of ~92.5% for a ResNet32 model on CIFAR-10.
    The default settings in the code example above are already quite optimized, thus we can expect at most an accuracy slightly above 96%.

Outputs

.. code-block:: text

    > Search is scheduled for 24:00:00 (h:m:s)
    > Best f: 0.965 (out of 98 params):  99%|█████████▉| [23:51:48<08:11, 14.6 min/param]
    > ========================== Summary =========================
    > Mode              : Best f : Steps : Time
    > ----------------  : ----   : ----  : ----
    > Initial solution  : 0.0968 : 1     : 01:50:21 (h:m:s)
    > Random seeding    : 0.96   : 23    : 1 day 18:22:52 (h:m:s)
    > Local sampling    : 0.965  : 74    : 6 days 00:59:26 (h:m:s)
    > ----------------  : ----   : ----  : ----
    > Total             : 0.965  : 98    : 23:51:48 (h:m:s)
    > ============================================================

    > Tuned params: Test accuracy = 96.2

.. note::

    Reproducing these numbers require 8 GPUs and 24h of runtime