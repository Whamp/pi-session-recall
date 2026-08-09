# Native user-scheduler contract

> The single-job cadence in this investigation is superseded by [Split scheduled indexing from zvec optimization](../adr/0008-split-scheduled-indexing-from-zvec-optimization.md). Its interval validation, native lifecycle, path capture, and platform constraints still apply.

## Decision

Conversation Recall will expose only:

```text
psr auto-index install [--interval <N>m|<N>h]
psr auto-index uninstall
```

The default interval is `1h`. An explicit interval must match `[1-9][0-9]*[mh]`: a positive base-10 whole number followed by lowercase `m` or `h`. Reject zero, signs, fractions, whitespace, missing or uppercase units, and every other duration form. Do not add status, notification, retry, watcher, daemon, or other lifecycle commands. This is the interface requested by [Ship scheduled automatic recall index maintenance](https://github.com/Whamp/pi-session-recall/issues/153) and investigated by [Verify native user-scheduler behavior](https://github.com/Whamp/pi-session-recall/issues/154).

Scheduling is opt-in, per-user, and never uses `sudo`. The scheduled operation is exactly `psr index`. It remains the sole writer and change detector required by [Keep whole-session recall maintenance outside interactive Pi](../adr/0003-keep-whole-session-recall-maintenance-explicit.md); scheduler code must not scan sessions, write index state, or add locks.

## Shared invocation contract

Installations capture absolute paths to the current Node executable, package root, and `bin/psr`, then invoke:

```text
<absolute-node> --import tsx <absolute-package-root>/bin/psr index
```

The package root is the working directory. The generated definition must escape paths for its native format and must not depend on a shell, shebang lookup, or interactive `PATH`. This follows systemd's command-line rules, which require an absolute executable or a name from systemd's fixed search path and do not interpret an `ExecStart=` line as a shell command ([`systemd.service` command lines](https://www.freedesktop.org/software/systemd/man/latest/systemd.service.html#Command%20Lines)). Apple's `ProgramArguments` maps directly to `execvp` arguments ([Apple `launchd.plist(5)`](https://github.com/apple-oss-distributions/launchd/blob/d448a1c8f70a61202f8705f94337f686b87c30c4/man/launchd.plist.5#L145-L155)).

Install the durable schedule before the immediate attempt. Reinstall replaces the existing definition and refreshes all captured paths. Generated definitions do not copy `PI_RECALL_*` variables from the installation shell: scheduled runs use the durable recall configuration file and normal defaults implemented by [`loadRecallConversationConfig`](../../src/recall-conversation-config.ts).

A scheduler-installation failure is fatal. An indexing failure does not remove or disable an installed schedule; the next interval supplies the retry. Do not add scheduler-specific state, retry, or locking.

## Linux: systemd user units

systemd's user-manager load path is `${XDG_CONFIG_HOME:-$HOME/.config}/systemd/user/` ([`systemd.unit` user load path](https://www.freedesktop.org/software/systemd/man/latest/systemd.unit.html#Unit%20File%20Load%20Path)). Write these stable files there:

```text
pi-session-recall-index.service
pi-session-recall-index.timer
```

### Service

```ini
[Unit]
Description=Maintain the pi-session-recall index

[Service]
Type=oneshot
WorkingDirectory=<absolute-package-root>
ExecStart=<absolute-node> --import tsx <absolute-package-root>/bin/psr index
StandardOutput=journal
StandardError=journal
```

Do not set `Restart=` or `RemainAfterExit=`. A oneshot without `RemainAfterExit=` returns to an inactive or failed state after it exits, making it suitable for repeated timer activation ([`systemd.service` `Type=`](https://www.freedesktop.org/software/systemd/man/latest/systemd.service.html#Type=)). Explicit journal destinations avoid depending on manager defaults ([`systemd.exec` standard output](https://www.freedesktop.org/software/systemd/man/latest/systemd.exec.html#StandardOutput=)).

### Timer

```ini
[Unit]
Description=Schedule pi-session-recall index maintenance

[Timer]
OnActiveSec=<normalized-interval>
OnUnitActiveSec=<normalized-interval>

[Install]
WantedBy=timers.target
```

Render minutes as `Nmin`; hours may remain `Nh`. `OnActiveSec=` schedules from timer activation and `OnUnitActiveSec=` schedules from the service's last activation ([`systemd.timer` monotonic timers](https://www.freedesktop.org/software/systemd/man/latest/systemd.timer.html#OnActiveSec=), [`systemd.time` time spans](https://www.freedesktop.org/software/systemd/man/latest/systemd.time.html#Parsing%20Time%20Spans)).

Do not add `Persistent=`, `OnCalendar=`, or custom retry behavior. Keep systemd's default timer accuracy; this feature does not require second-level precision. If the service is active when the timer elapses, systemd leaves that instance running instead of spawning an overlapping instance ([`systemd.timer` description](https://www.freedesktop.org/software/systemd/man/latest/systemd.timer.html#Description)). This protects only starts through this unit, not a separately invoked `psr index` process.

### Install and reinstall

After validating the interval and resolving absolute paths:

1. Write or replace both unit files.
2. Run `systemctl --user daemon-reload`.
3. Run `systemctl --user enable pi-session-recall-index.timer`.
4. Run `systemctl --user restart pi-session-recall-index.timer`.
5. Run `systemctl --user start pi-session-recall-index.service`.

`enable` creates the `[Install]` symlink but does not start the timer, while `restart` also starts an inactive timer ([`systemctl enable`](https://www.freedesktop.org/software/systemd/man/latest/systemctl.html#enable%20UNIT%E2%80%A6), [`systemctl restart`](https://www.freedesktop.org/software/systemd/man/latest/systemctl.html#restart%20PATTERN%E2%80%A6)). Failures in steps 1–4 fail installation. If the immediate oneshot in step 5 fails, emit a warning and report installation success; the timer is already installed and active.

### Uninstall and logs

Run, in order:

```bash
systemctl --user disable --now pi-session-recall-index.timer
systemctl --user stop pi-session-recall-index.service
```

Then remove both files and run:

```bash
systemctl --user daemon-reload
```

Treat already-absent units and files as harmless. Stopping the service explicitly matters because disabling and stopping the timer does not stop a service it already launched ([`systemctl disable`](https://www.freedesktop.org/software/systemd/man/latest/systemctl.html#disable%20UNIT%E2%80%A6)). Do not add failed-state cleanup.

Read indexing logs with:

```bash
journalctl --user-unit=pi-session-recall-index.service
```

The filter is defined by [`journalctl --user-unit=`](https://www.freedesktop.org/software/systemd/man/latest/journalctl.html#--user-unit=).

## macOS: per-user LaunchAgent

Apple defines an agent as a per-user job and identifies `~/Library/LaunchAgents` as the user-provided agent directory ([Apple `launchd(8)`](https://github.com/apple-oss-distributions/launchd/blob/d448a1c8f70a61202f8705f94337f686b87c30c4/man/launchd.8#L43-L63), [Creating Launch Daemons and Agents](https://developer.apple.com/library/archive/documentation/MacOSX/Conceptual/BPSystemStartup/Chapters/CreatingLaunchdJobs.html)). Use:

```text
Label: dev.pi-session-recall.auto-index
Path:  ~/Library/LaunchAgents/dev.pi-session-recall.auto-index.plist
Owner: current user
Mode:  0600
```

Apple requires a per-user plist to be owned by its loading user and forbids group or world write access; mode `0600` is the simple stricter choice ([Apple `launchctl(1)`](https://github.com/apple-oss-distributions/launchd/blob/d448a1c8f70a61202f8705f94337f686b87c30c4/man/launchctl.1#L27-L39)). Create `~/.pi/agent/logs/` and generate a correctly XML-escaped plist containing:

```text
Label = dev.pi-session-recall.auto-index
ProgramArguments = [
  <absolute-node>,
  --import,
  tsx,
  <absolute-package-root>/bin/psr,
  index
]
WorkingDirectory = <absolute-package-root>
StartInterval = <interval-in-seconds>
RunAtLoad = true
StandardOutPath = ~/.pi/agent/logs/pi-session-recall-auto-index.out.log
StandardErrorPath = ~/.pi/agent/logs/pi-session-recall-auto-index.err.log
```

Convert minutes to `N × 60` and hours to `N × 3600`. Apple documents `RunAtLoad` as one launch when the job loads, `StartInterval` as a launch every N seconds, and the working-directory and standard-stream path keys directly ([Apple `launchd.plist(5)` keys](https://github.com/apple-oss-distributions/launchd/blob/d448a1c8f70a61202f8705f94337f686b87c30c4/man/launchd.plist.5#L210-L219), [interval and stream keys](https://github.com/apple-oss-distributions/launchd/blob/d448a1c8f70a61202f8705f94337f686b87c30c4/man/launchd.plist.5#L245-L283)). Apple's published launchd source opens configured stdout and stderr with `O_APPEND` ([`core.c`](https://github.com/apple-oss-distributions/launchd/blob/d448a1c8f70a61202f8705f94337f686b87c30c4/src/core.c#L5183-L5190)). Log rotation is out of scope.

Do not add `KeepAlive`, `LaunchOnlyOnce`, retries, rotation, or scheduler-specific locking. Apple's published source declines to dispatch an already-active job, which supports the no-overlap expectation ([`core.c`](https://github.com/apple-oss-distributions/launchd/blob/d448a1c8f70a61202f8705f94337f686b87c30c4/src/core.c#L3963-L3991)).

### Lifecycle

Use only the documented path lifecycle:

```bash
launchctl load <absolute-plist-path>
launchctl unload <absolute-plist-path>
```

Apple documents `load` for configuration files and `unload` for removing and stopping a running job ([Apple `launchctl(1)`](https://github.com/apple-oss-distributions/launchd/blob/d448a1c8f70a61202f8705f94337f686b87c30c4/man/launchctl.1#L27-L70)). Do not add `bootstrap gui/<uid>` or `bootout`: the available primary documentation and this Linux-only investigation did not establish or runtime-test that modern domain-target lifecycle.

Install or reinstall in this order:

1. Validate the interval and resolve absolute paths.
2. Attempt `launchctl unload <plist>`, tolerating an already-unloaded or absent job.
3. Write the plist with current-user ownership and mode `0600`.
4. Run `launchctl load <plist>`.

A load failure fails installation. `RunAtLoad=true` requests the immediate indexing attempt only after the durable plist exists. The documented `load` interface does not provide the later child-process result to this contract, so a later `psr index` failure appears in the configured error log and ordinary exit status; it does not roll back the job or make a completed load fail. Do not add polling or a wrapper merely to turn that asynchronous failure into an installer warning.

Uninstall by unloading the plist, tolerating an already-unloaded job, and then removing the plist, tolerating an absent file. `unload` also stops a running instance.

## Verification boundary

Linux requires unit-content tests, lifecycle-order tests, failure-path tests, and an actual user-timer smoke test. Platform-independent tests must cover interval validation and conversion, absolute captured paths, generated artifacts and prohibited keys, replacement and absent-state behavior, mode `0600`, Linux's nonfatal immediate failure, and macOS `RunAtLoad=true`.

No Mac was available. The following claims remain **runtime-untested on macOS** and must be labeled that way in implementation comments or user documentation:

- the target macOS `launchctl` accepts the generated plist and selected `load`/`unload` lifecycle;
- `RunAtLoad` starts `psr index` immediately;
- `StartInterval` continues after an exit-status-1 run;
- launchd suppresses overlapping runs as expected;
- absolute Node plus `--import tsx` works in the LaunchAgent environment;
- stdout and stderr append to the configured files;
- `psr index`, its durable recall configuration, and its embedding endpoint work in that environment.

There is no accepted behavior for Windows, status reporting, scheduler migration, notifications, custom retries, or log rotation.
