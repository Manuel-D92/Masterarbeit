# Hockey GLT Demo Software

This is the goal line technology demo software developed as an alternative to the PyDfw based software stack.

The Hockey GLT software features interfaces for the following:
 - Calibration (same format as used by the PyDfw-based software stack)
 - Debugging (view signals of one antenna at one frequency at a time w/ or w/o calibration)
 - Goal Detection (displays DDFT signals used for goal detection and features a big alert bar, flashing when goals are detected)
 
## Requirements

This software is (in contrast to the PyDfw-based one) written in Python 3 and thus requires Python 3.5 to be installed on target computers.

Additional dependencies are...
 - __numpy__
 - __matplotlib__

## Why has this been rewritten?

The primary reason for the reimplementation of this system is to be independent of the PyDfw framework and to be able to distribute copies of the software without the need for including PyDfw.

As a side-effect the installation has been greatly simplified (because of reduced dependencies).

Nevertheless, the main advantage of the PyDfw-based solution remains in quicker interaction with the system via bypassing the restricted GUI and using MCTool. This is (obviously) not available in this version.

## Branches description:

- __master:__ Contains a Goal detector, no localisation, calibration, debug and a tab to show the noise of the signals

- __3D-localisation:__ Just an old version of the 3D localisation

- __coburg:__ Here is the version we used at Audi. Contains 2D localisation with 4 positions and an automatic offset calibration.

- __HMI__ and __HMI_Executables:__ That is the version for the HMI. Contains the game, 3D localisation, 2D localisation and LED control.

- __SCS:__ Version for SCS. Contains 3D localisation.

- __Frontend:__ Hansenjo

- __rfid:__ RFID correlation localization scripts in two variants
