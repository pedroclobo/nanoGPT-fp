# Render assets/bigram.gif — run from the repo root:  vhs assets/bigram.tape
Output assets/bigram.gif

Set FontSize 16
Set Width 1000
Set Height 600
Set Padding 20
Set WaitTimeout 120s

Type ".venv/bin/python bigram.py --generate checkpoints/bigram_CS8.pt --num_tokens 10000"
Enter
Wait
