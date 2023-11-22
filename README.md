## Env
set environment variables
```
echo 'export $(cat .env | grep -v ^#)' >> ~/.bashrc
```

install gcsfuse & mount bucket
```
source gcp/install_gcsfuse.sh
source gcp/mount_bucket.sh
```

## Rye
```
curl -sSf https://rye-up.com/get | bash
echo 'source "$HOME/.rye/env"' >> ~/.bashrc
```

## Bitstandbytes
https://github.com/TimDettmers/bitsandbytes/issues/620#issuecomment-1666014197

`.venv/lib/python3.10/site-packages/bitsandbytes/cuda_setup/env_vars.py`
```
def to_be_ignored(env_var: str, value: str) -> bool:
    ignorable = {
        "PWD",  # PWD: this is how the shell keeps track of the current working dir
        "OLDPWD",
        "SSH_AUTH_SOCK",  # SSH stuff, therefore unrelated
        "SSH_TTY",
        "HOME",  # Linux shell default
        "TMUX",  # Terminal Multiplexer
        "XDG_DATA_DIRS",  # XDG: Desktop environment stuff
        "XDG_GREETER_DATA_DIR",  # XDG: Desktop environment stuff
        "XDG_RUNTIME_DIR",
        "MAIL",  # something related to emails
        "SHELL",  # binary for currently invoked shell
        "DBUS_SESSION_BUS_ADDRESS",  # hardware related
        "PATH",  # this is for finding binaries, not libraries
        "LESSOPEN",  # related to the `less` command
        "LESSCLOSE",
        "GOOGLE_VM_CONFIG_LOCK_FILE",  # <----------------------- add 
        "_",  # current Python interpreter
    }
    return env_var in ignorable
```