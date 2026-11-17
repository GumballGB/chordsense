--**--**-- WELCOME TO CHORDSENSE --**--**--

The goal of this repository is to make something cool out music and machine learning

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

DEVELOPER NOTES

For the complete pipeline, remember these!!

1. RECORD: Find some way to record the audio

2. MONO: The audio is most likely to be stereo, which is 2-channel.
Our stack does not process that, so it should be converted for MONO

3. NORMALIZE: The amplitude of some signal will probably be higher than int's range.
The range of int is 32k. You actually faced a similar problem in your guitar tuner project (ECSE444).
Where all signals were just clipped to 32k all the time. So normalize it to be between -1 and 1.
Also remember you had an issue with negative values for amplitude, it sort of just became positive

4. FRAME: Only process FFT under a certain time window at a time, because chord progressions exist.
Study the Hann Window, it is useful somehow. 

5. FFT: The math is severely undervalued with np.fft 

6. CHROMA: Translate frequency bank into a bank of notes, it will lead to an array(like) of 12

7. TEMPLATE MATCHING: Compare that array(like) of notes to a template, to see which template it matches the most
This is where chord detection probably happens.

8. SMOOTHING: A filter that removes flickering chords over time.

As of now, you're still just doing one note at a time. 

