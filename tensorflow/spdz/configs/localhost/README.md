
# Localhost configuration

This is the easiest way to get started since very little setup is required. But since all players are running on the same machine we of course don't have the desired security properties nor accurate performance measures.

## Setup

To use this configuration we first specify that we wish to it by symlinking `config.py` and `run.sh` to the root directory:

```sh
spdz $ ln -s configs/localhost/config.py
spdz $ ln -s configs/localhost/run.sh
```

and simply start the five local processes:

```sh
spdz $ cd configs/localhost/
localhost $ ./start.sh
```

This will block the current terminal, so to actually execute the code we need another terminal.

## Running

Once setup simply execute

```sh
spdz $ ./run.sh <file.py>
```

## Cleanup

A script is also provided to stop the processes again (but be a bit careful as it's currently not doing so in a particularly nice way):

```sh
spdz $ cd configs/localhost/
localhost $ ./stop.sh
```

At this point it's safe to delete the symlink in case you e.g. wish to experiment with another configuration:

```sh
spdz $ rm config.py \
       rm run.sh
```
