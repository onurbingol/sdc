# RFD: SD Challenge

- Authors: Onur Bingol
- Status: Draft
- Date: [2025-06-09]
- RFD #: 001

## Summary

This application is designed to simulate a ballistic missile striking a distant target using the simulation software, [Elodin](https://docs.elodin.systems/home/quickstart/).

## Motivation

Simulations are designed to give users an idea on the application, so that they could understand the it, define the principles and processes, and test the system or components of the application in a virtual environment. Ability to create a simulation environment may help the users test their products, systems or applications without going through expensive and time-consuming manufacturing
processes, hence reducing the waste of materials and resources for many applications. For defense applications, it is also required to find a distant location to physically test the systems due to the nature of the applications. Using a simulation software would not completely remove the manufacturing or physical testing requirements, but it could help you to make better choices with less trials.

## Guide-level explanation

Please follow the instructions on the [README.md](README.md) file to install and start the application via the command-line client. The simulation can be configured by the global variables included at the top of the [sdc_main.py](src/sdc/sdc_main.py) file as of current application version.

## Reference-level explanation

* Currently, the application focuses on launching the missile by setting the elevation and azimuth angles.
* The geometry of the missile is a ball for simplicity, as the development started from the [bouncing ball](https://docs.elodin.systems/home/bouncing-ball/) example provided by [Elodin](https://docs.elodin.systems/home/quickstart/).
* The application reads the thrust curve of [AeroTecg M685W](https://www.thrustcurve.org/motors/AeroTech/M685W/) by default. This is hard-coded in the application, but can be swapped to use a different thrust curve.
* The `bounciness` variable in the [bouncing ball](https://docs.elodin.systems/home/bouncing-ball/) example is removed to create a rigid ground plane.
* Wind drag is kept the same as the [bouncing ball](https://docs.elodin.systems/home/bouncing-ball/) example and it depends on the ball radius.
* Applying thrust, gravity and wind drag should be added to the effectors section of the simulator, e.g. `elodin.six_dof(sys=<here>)`, as forces are computed by integration.
* Data generation functions should be passed via a pipe (`|`) to `elodin.six_dof`, e.g. `compute_thrust | elodin.six_dof(sys=apply_thrust)` where `compute_thrust` computes the thrust direction vector from the input elevation and azimuth angles, interpolates the current thrust vector magnitude from the input thrust curve, and creates the thrust vector by `magnitude * direction`.

## Drawbacks

One of the biggest drawbacks is the simulation configuration setup. As of this release, it needs to be manually configured by changing the global variables in [sdc_main.py](src/sdc/sdc_main.py).

The capabilities of this application is also limited as of this version:

* No automated computation of elevation and azimuth angles to strike a target defined by its coordinates.
* No dynamic computation of elevation and azimuth angles to strike a moving target. The target may also move in a varying speed.
* No wind direction or wind magnitude configuration. The wind direction and magnitude may also change during the simulation run.

## Future Possibilities

* Automatically compute the viewport variables to get the full view of missile launching and hitting the target.
* Dynamic updates to the wiewport, e.g. follow the missile and follow the target cameras.

## Appendix

* [Elodin GitHub repository](https://github.com/elodin-sys/elodin)
* [Elodin documentation](https://docs.elodin.systems/home/quickstart/)
* [Elodin examples](https://github.com/elodin-sys/elodin/tree/main/examples)
