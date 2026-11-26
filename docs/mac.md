# The Mac app

The modelmap app shows the map of your models in its own window and
removes the ones you choose. It is `modelmap serve` with a Mac window
around it: the scan, the map and every removal are the same code the
command line runs, from a Python bundled inside the app, so nothing else
needs to be installed.

## Get the app

The app is built from a clone of the repository. It needs Xcode,
[XcodeGen](https://github.com/yonaskolb/XcodeGen) (`brew install xcodegen`),
uv and Poetry.

```bash
git clone https://github.com/Flux-Frontiers/modelmap
cd modelmap/macos
make run
```

`make run` builds `macos/build/Build/Products/Release/modelmap.app` and
opens it. Drag it to Applications to keep it. The build is for Apple
silicon, needs macOS 15 or later, and is signed to run on the Mac that
built it.

## Using it

When the app opens, it scans for models, which takes a few seconds, then
shows the map:

- Each block is a store (Hugging Face, Ollama, oMLX, loose files and so
  on) and each tile a model, sized by the space it takes.
- Click a tile to open the model: the square is redrawn as its files, and
  the panel beside it lists each file with its full path.
- Go back up with Escape or the trail above the map.
- Group by store, format, family or kind, and filter by name, with the
  controls above the map.
- The panel lists every entry with a problem, such as an incomplete
  download, even when its tile is too small to see.

## Choosing what to scan

By default the app reads every model store it knows and sweeps your home
folder. To scan somewhere else, such as an external drive or a folder of
downloads, choose Map > Choose Folders to Scan (Cmd-O) and pick one or
more folders. The **Scan** menu at the bottom of the panel says what to
scan with them:

- **These folders, the model stores and the home folder**: everything the
  default scan covers, plus your folders.
- **These folders and the model stores**: the stores and your folders,
  without the sweep of your home folder.
- **Only these folders**: nothing else. This is the choice the panel
  starts on, until you make another.

The window's title shows what is being scanned, and the app remembers the
choice the next time it opens. Map > Scan Usual Locations goes back to
the default.

## Removing a model

Open the model and press **Remove** in its panel. A dialog lists exactly
what will happen:

- for an Ollama model, the `ollama rm` command, with every tag that shares
  its weights;
- for a Hugging Face cache repository, the `hf cache rm` command;
- for anything else, every file or folder that will move to the Trash.

It also says whether the removal can be undone. Files moved to the Trash
can be put back from there; what `ollama rm` and `hf cache rm` remove
cannot. Nothing happens until you press **Remove** in the dialog. Cancel
has the focus, so pressing Return cancels.

When you confirm, the app scans again before acting. If anything changed
since the dialog opened, for example a file was added to the model's
folder, it removes nothing and shows the new list for you to check.
After a removal the map is redrawn without the model.

A model has no Remove button when removing it would do harm: a cache that
its tool rebuilds, files that nothing refers to, or files another model
also uses. The panel says why. The [guide](guide.md#removing-models)
explains each rule.

## Menus

| Menu item | Shortcut | Does |
|-----------|----------|------|
| Map > Choose Folders to Scan | Cmd-O | Scan folders or drives you pick, with or without the stores and the home folder |
| Map > Scan Usual Locations | | Go back to the stores and the home folder |
| Map > Rescan | Cmd-R | Scan again and redraw the map |
| Map > Reload Page | Shift-Cmd-R | Redraw the page without scanning |
| Map > Open in Browser | Cmd-B | Open the same page in your web browser |
| Map > Restart Server | | Stop and start the server, then scan |
| Window > Server Log | Cmd-L | Every request and every removal, with the command it ran |
| Help > modelmap Help | Cmd-? | Help inside the app |

## Where it looks

The app looks where the command line does: the stores of Hugging Face,
Ollama, oMLX, LM Studio, llama.cpp, GPT4All, Jan and Whisper, and your
home folder for loose model files. It reads your login shell's
environment, so settings in your shell profile that move a store, such as
`HF_HOME` or `OLLAMA_MODELS`, apply, and the `ollama` and `hf` commands
are found where your shell finds them.

## Folder access

The first scan sweeps your home folder, and macOS asks before an app reads
Desktop, Documents or Downloads. Choose **Allow** for modelmap to find
model files there; choose **Don't Allow** and it skips that folder. You
can change the answer later in System
Settings > Privacy & Security > Files and Folders.

A build signed only for its own Mac counts as a new app each time it is
rebuilt, so macOS asks again after every rebuild.

## Privacy

Nothing leaves your Mac. The app's server listens only on `127.0.0.1`,
needs a token made for each run, and answers only its own page, so no web
page you have open elsewhere can read the map or remove anything. The
Server Log hides the token.

## Troubleshooting

**A store is missing.** If you moved a store with a setting such as
`HF_HOME`, set it in your shell profile (`~/.zprofile` or
`~/.bash_profile`), then choose Map > Restart Server. The app reads the
profile when the server starts. To see where each store is looked for,
run `modelmap providers` in Terminal.

**Remove says a command is not found.** Removing an Ollama model needs
`ollama`, and a Hugging Face cache repository needs `hf` (from
`huggingface_hub`). Install the one named, make sure your shell finds it,
then choose Map > Restart Server.

**Remove says Ollama is not running.** `ollama rm` acts on the running
Ollama. Start Ollama, then try again.

**"modelmap could not start."** The window shows the end of the server's
output. Press **Try Again**, and check Window > Server Log for the cause.

**The app is not sandboxed.** modelmap reads model stores across your
home folder and other drives, runs `ollama`, `hf` and `trash`, and moves
files to the Trash, none of which the App Sandbox allows. For the same
reason the app is not on the Mac App Store.
