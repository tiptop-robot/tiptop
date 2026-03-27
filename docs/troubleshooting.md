# Troubleshooting

This page provides solutions to common issues you may encounter when setting up and running TiPToP.

## ZED Camera not detected

- Make sure no other scripts or processes are already using the ZED Camera.
- Verify the ZED camera is powered and connected
- Try replugging both ends of the USB cable. On the USB-C connector on the camera side, try plugging it in both directions.
- Check the camera serial number in `tiptop/config/tiptop.yml` matches your hardware
- Run `viz-gripper-cam` to test the camera feed

## Cannot connect to Franka robot

- Ensure the Bamboo controller is running on the control workstation
- Verify network connectivity: `ping <robot-hostname>`
- Check the robot URL in `tiptop/config/tiptop.yml` matches your control workstation address
- Confirm the robot is not in a fault state (check Franka Desk)

## Robotiq gripper does not work

- Verify the gripper server is running on the control workstation with Bamboo
- Check the user is in the `tty` and `dialout` groups (see [Installation](installation.md#installing-bamboo))
- Test manually with `gripper-open` and `gripper-close` commands

## cuTAMP/cuRobo motion planning fails

- Verify workspace obstacles are correctly defined
- Check calibration with `viz-calibration`
- Ensure objects are within the robot's reachable workspace
- Try reducing `time_dilation_factor` if trajectory execution is too fast

## Perception issues

- Confirm FoundationStereo and M2T2 servers are running
- Check service URLs in `tiptop/config/tiptop.yml`
- Verify the capture pose provides good workspace coverage
- Test depth estimation quality with `viz-scene`

## libfranka realtime scheduling error

If you encounter `libfranka: unable to set realtime scheduling: Operation not permitted`, your user needs permission to set realtime priorities.

**Check groups and add user to realtime group:**

```bash
groups  # Check if 'realtime' is listed
sudo usermod -aG realtime $USER
```

**Configure realtime limits:**

Edit `/etc/security/limits.conf` and add the following lines:

```
@realtime soft rtprio 99
@realtime soft priority 99
@realtime soft memlock 102400
@realtime hard rtprio 99
@realtime hard priority 99
@realtime hard memlock 102400
```

Log out and back in to apply the changes. Verify with `ulimit -r` (should show 99).
