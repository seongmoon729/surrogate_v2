from pathlib import Path
import torch


class Checkpoint:
    def __init__(self, path):
        self.base_path = Path(path)
        self.base_path.mkdir(parents=True, exist_ok=True)

    def save(self, network, optimizer=None, scheduler=None, step=0, persistent_period=None):
        ckpt_path = self.base_path / f"checkpoint_{step}.pth"

        # Choose variables to save.
        target_objects = {'network': network.state_dict(), 'step': step}
        if optimizer:
            target_objects.update({'optimizer': optimizer.state_dict()})
        if scheduler:
            target_objects.update({'scheduler': scheduler.state_dict()})

        # Save.
        torch.save(target_objects, ckpt_path)
        
        # Delete old checkpoints except persistent checkpoints.
        old_ckpt_paths = set(self.base_path.glob('*.pth')) - set([ckpt_path])
        for old_ckpt_path in old_ckpt_paths:
            if self._get_step(old_ckpt_path) % persistent_period:
                old_ckpt_path.unlink()

    def load(self, network, optimizer=None, scheduler=None, step=-1):
        ckpt_paths = sorted(self.base_path.glob('*.pth'), key=self._get_step)

        if len(ckpt_paths):
            if step == -1:
                target_ckpt_path = ckpt_paths[step]
            else:
                target_ckpt_path = self.base_path / f"checkpoint_{step}.pth"
                assert target_ckpt_path.exists()
            
            # Load.
            target_ckpt = torch.load(target_ckpt_path, map_location=network.device)
            if 'network' in target_ckpt:
                network.load_state_dict(target_ckpt['network'])
            elif 'model' in target_ckpt:
                network.load_state_dict(target_ckpt['model'])
            if optimizer:
                optimizer.load_state_dict(target_ckpt['optimizer'])
            if scheduler:
                scheduler.load_state_dict(scheduler['scheduler'])
            step = target_ckpt['step']
        else:
            step = 0
        return step

    def resume(self, network, optimizer, step=-1):
        step = self.load(network, optimizer, step)
        return step

    def _get_step(self, ckpt_path):
        step = int(ckpt_path.stem.split('_')[-1])
        return step
