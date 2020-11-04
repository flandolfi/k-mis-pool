from datetime import timedelta, datetime
from skorch.callbacks import Callback


class LateStopping(Callback):
    def __init__(self, sink=print, **time_kwargs):
        self.timedelta = timedelta(**time_kwargs)
        self.sink = sink
        self.start = None

    def on_train_begin(self, *args, **kwargs):
        self.start = datetime.now()

    def on_epoch_end(self, *args, **kwargs):
        delta = datetime.now() - self.start

        if delta > self.timedelta:
            self._sink(f"Training timeout: {delta} > {self.timedelta}.", net.verbose)
            raise KeyboardInterrupt

    def _sink(self, text, verbose):
        if (self.sink is not print) or verbose:
            self.sink(text)


class LRLowerBound(Callback):
    def __init__(self, min_lr=1e-6, sink=print):
        self.min_lr = min_lr
        self.sink = sink

    def on_epoch_end(self, net, *args, **kwargs):
        lr = net.optimizer_.param_groups[0]['lr']
        net.history.record('learning_rate', lr)

        if lr < self.min_lr:
            self._sink(f"Stopping training after reaching LR lower-bound: "
                       f"{lr} < {self.min_lr}.", net.verbose)
            raise KeyboardInterrupt

    def _sink(self, text, verbose):
        if (self.sink is not print) or verbose:
            self.sink(text)
