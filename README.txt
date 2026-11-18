--**--**-- WELCOME TO CHORDSENSE --**--**--

As of now, this project is DSP stack that specializes in the detection of chords.
It can also detect chord progressions and perform live chord analysis.

The goal in the future is to bring some sort of Machine Learning into this stack.

Like all python stacks it requires its own environment.

--- FIRST SETUP ---
To setup this project for first-time users, you need to run the following commands;

1. Create the venv (virtual environment), the command is:

python3.12 -m venv .venv

If on LINUX, install the python venv w/ this line (it works on my computer)
sudo apt install python3-venv python3-pip

You also need to capture audio on a linux machine
sudo apt install libportaudio2

Then

python3.12 -m venv .venv --prompt chordsense

2. Activate the virtual environment with:

source .venv/bin/activate

3. Install the dependencies, it is found in a requirements.txt.
Make sure you install these dependencies while in the environment

pip install -r requirements.txt

----

