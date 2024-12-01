import sys
import random
import shutil

import os

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import torch
import torch.optim as optim

from loss import RateDistortionLoss
from model import Net

from utils import (
    get_logger,
    parse_args,
    prepare_data,
    CustomDataParallel,
    configure_optimizers,
    AverageMeter,
)


logger = get_logger()


def main(argv):
    args = parse_args(argv)
    if args.seed is not None:
        torch.manual_seed(args.seed)
        random.seed(args.seed)

    device = "cuda" if args.cuda and torch.cuda.is_available() else "cpu"

    train_dataloader, test_dataloader = prepare_data(args, device)

    net = Net()
    net = net.to(device)
    if args.cuda and torch.cuda.device_count() > 1:
        net = CustomDataParallel(net)

    optimizer, aux_optimizer = configure_optimizers(net, args)
    lr_scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer)
    criterion = RateDistortionLoss(lmbda=args.lmbda)

    last_epoch = 0
    if args.checkpoint:
        logger.info(f"loading {args.checkpoint}")
        checkpoint = torch.load(args.checkpoint, map_location=device)
        last_epoch = checkpoint["epoch"] + 1
        net.load_state_dict(checkpoint["state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        aux_optimizer.load_state_dict(checkpoint["aux_optimizer"])
        lr_scheduler.load_state_dict(checkpoint["lr_scheduler"])

    best_loss = float("inf")
    for epoch in range(last_epoch, args.epochs):
        logger.info(f"Learning rate: {optimizer.param_groups[0]['lr']}")
        train_one_epoch(
            net,
            criterion,
            train_dataloader,
            optimizer,
            aux_optimizer,
            epoch,
            args.clip_max_norm,
        )
        loss = test_epoch(epoch, test_dataloader, net, criterion)
        lr_scheduler.step(loss)

        is_best = loss < best_loss
        best_loss = min(loss, best_loss)

        if args.save:
            torch.save(
                {
                    "epoch": epoch,
                    "state_dict": net.state_dict(),
                    "loss": loss,
                    "optimizer": optimizer.state_dict(),
                    "aux_optimizer": aux_optimizer.state_dict(),
                    "lr_scheduler": lr_scheduler.state_dict(),
                },
                "checkpoint.pth.tar",
            )
            if is_best:
                shutil.copyfile("checkpoint.pth.tar", "checkpoint_best_loss.pth.tar")


def train_one_epoch(
    model, criterion, train_dataloader, optimizer, aux_optimizer, epoch, clip_max_norm
):
    model.train()
    device = next(model.parameters()).device

    log_points = iter([0, 25, 50, 75, 100, 101])
    log_point = next(log_points)
    for i, d in enumerate(train_dataloader):
        d = d.to(device)

        optimizer.zero_grad()
        aux_optimizer.zero_grad()

        loss = 0 if random.random() < 0.8 else random.choice([i / 100 for i in range(10, 70, 10)])
        out_net = model(d, loss)

        out_criterion = criterion(out_net, d)
        out_criterion["loss"].backward()
        if clip_max_norm > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), clip_max_norm)
        optimizer.step()

        aux_loss = model.aux_loss()
        aux_loss.backward()
        aux_optimizer.step()

        percentage = round(100.0 * i / len(train_dataloader))
        if percentage >= log_point:
            logger.info(
                f"Train epoch {epoch}: ["
                f"{i*len(d)}/{len(train_dataloader.dataset)}"
                f"({percentage}%)]"
                f"\tLoss: {out_criterion['loss'].item():.3f} |"
                f"\tMSE loss: {out_criterion['mse_loss'].item():.3f} |"
                f"\tBpp loss: {out_criterion['bpp_loss'].item():.2f} |"
                f"\tAux loss: {aux_loss.item():.2f}"
            )
            log_point = next(log_points)


def test_epoch(epoch, test_dataloader, model, criterion):
    model.eval()
    device = next(model.parameters()).device

    loss = AverageMeter()
    bpp_loss = AverageMeter()
    mse_loss = AverageMeter()
    aux_loss = AverageMeter()

    with torch.no_grad():
        for d in test_dataloader:
            d = d.to(device)
            pkt_loss = 0 if random.random() < 0.8 else random.choice([i / 100 for i in range(10, 70, 10)])
            out_net = model(d, pkt_loss)
            out_criterion = criterion(out_net, d)

            aux_loss.update(model.aux_loss())
            bpp_loss.update(out_criterion["bpp_loss"])
            loss.update(out_criterion["loss"])
            mse_loss.update(out_criterion["mse_loss"])

    logger.info(
        f"Test epoch {epoch}: Average losses:"
        f"\tLoss: {loss.avg:.3f} |"
        f"\tMSE loss: {mse_loss.avg:.3f} |"
        f"\tBpp loss: {bpp_loss.avg:.2f} |"
        f"\tAux loss: {aux_loss.avg:.2f}\n"
    )

    return loss.avg


if __name__ == "__main__":
    main(sys.argv[1:])
