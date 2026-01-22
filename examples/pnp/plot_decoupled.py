from registry import register_custom_arch_configs, register_custom_loss_configs, Parameters
import os
from matplotlib import pyplot as plt
import torch
import matplotlib
matplotlib.use("Agg")

from physicsnemo.sym.hydra import to_absolute_path, instantiate_arch, PhysicsNeMoConfig
from physicsnemo.sym.hydra.utils import compose, register_amp_configs
from physicsnemo.sym.key import Key


def plot_spatial_profiles(
        cfg: PhysicsNeMoConfig,
        input_dir: str,
        output_dir: str,
        nx=4000,
        checkpoint_dir=None
) -> None:

    # load checkpoint
    arch_cfg = cfg.arch[next(iter(cfg.arch))]
    cp_net = instantiate_arch(
        input_keys=[Key("x"), Key("y")],
        output_keys=[Key("cp")],
        cfg=arch_cfg,
    )
    cn_net = instantiate_arch(
        input_keys=[Key("x"), Key("y")],
        output_keys=[Key("cn")],
        cfg=arch_cfg,
    )
    phi_net = instantiate_arch(
        input_keys=[Key("x"), Key("y")],
        output_keys=[Key("phi")],
        cfg=arch_cfg,
    )
    cp_net.make_node(name='cp_net')
    cn_net.make_node(name='cn_net')
    phi_net.make_node(name='phi_net')

    # ensure checkpoint directory exists
    if checkpoint_dir is None:
        checkpoint_dir = to_absolute_path(f"./outputs/{input_dir}/")
    if not os.path.isdir(checkpoint_dir):
        raise ValueError(f"Checkpoint directory {checkpoint_dir} does not exist")

    # ensure checkpoint files loaded correctly
    try:
        cp_net.load(checkpoint_dir)
        cn_net.load(checkpoint_dir)
        phi_net.load(checkpoint_dir)
    except Exception as e:
        raise RuntimeError(
            f"Failed to load checkpoint for {next(iter(cfg.arch))} from {checkpoint_dir}: {e}"
        )
    cp_net.eval()
    cn_net.eval()
    phi_net.eval()

    # make figure
    fig, axs = plt.subplots(1, 3, figsize=(15, 5), dpi=200)
    axs[0].set_title("Cation Concentration")
    axs[1].set_title("Anion Concentration")
    axs[2].set_title("Electric Potential")
    for ax in axs:
        ax.set_box_aspect(1)

    # loop through time values
    p = Parameters()
    times = [0.0, 1.0, 6.0, 36.0, 100.0, 3600.0]

    for t in times:
        # create uniform input
        x = torch.linspace(0.0, 1.0, nx).reshape(-1, 1)
        invar = {
            "x": x,
            "y": torch.full_like(x, fill_value=t/p.t_c),
        }

        # unpack predictions
        cp_pred = cp_net(invar)
        cn_pred = cn_net(invar)
        phi_pred = phi_net(invar)
        cp_pred = cp_pred['cp'].clone().detach().numpy()
        cn_pred = cn_pred['cn'].clone().detach().numpy()
        phi_pred = phi_pred['phi'].clone().detach().numpy()

        # add plots
        axs[0].plot(x, cp_pred, label=f"t={t:.2f}s")
        axs[1].plot(x, cn_pred, label=f"t={t:.2f}s")
        axs[2].plot(x, phi_pred, label=f"t={t:.2f}s")

    # make plot
    for ax in axs:
        ax.legend(framealpha=0.0)
    plot_dir = to_absolute_path(f"./outputs/{output_dir}")
    os.makedirs(plot_dir, exist_ok=True)
    plt.savefig(plot_dir + "/plot.png")
    plt.close()


if __name__ == "__main__":
    register_amp_configs()
    register_custom_arch_configs()
    register_custom_loss_configs()

    cfg = compose(
        config_path="conf",
        config_name=f"config_fb"
    )

    plot_spatial_profiles(
        input_dir="pnp_fb",
        output_dir="plots",
        cfg=cfg,
    )