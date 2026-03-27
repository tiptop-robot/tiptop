"""Interactive configuration script for tiptop.yml"""

from pathlib import Path

from ruamel.yaml import YAML


def prompt_with_default(prompt_text: str, current_value: str, allow_skip: bool = True) -> str:
    """Prompt user for input with current value as default."""
    if current_value and allow_skip:
        display_value = current_value
        full_prompt = f"{prompt_text} [{display_value}]: "
    else:
        full_prompt = f"{prompt_text}: "

    user_input = input(full_prompt).strip()

    # If user just hits enter
    if not user_input:
        if allow_skip and current_value:
            return current_value
        # For required fields without defaults, keep prompting
        if not allow_skip or not current_value:
            print("  ⚠️  This field is required. Please enter a value.")
            return prompt_with_default(prompt_text, current_value, allow_skip)

    return user_input


_ROBOT_EMBODIMENTS = [
    {
        "name": "fr3_robotiq",
        "robot": "Franka FR3",
        "gripper": "Robotiq 2F-85",
        "wrist_camera_spheres": "Included (DROID setup)",
    },
    {
        "name": "panda_robotiq",
        "robot": "Franka Panda",
        "gripper": "Robotiq 2F-85",
        "wrist_camera_spheres": "Included (DROID setup)",
    },
    {
        "name": "fr3",
        "robot": "Franka FR3",
        "gripper": "Franka default hand",
        "wrist_camera_spheres": "Not modeled",
    },
    {
        "name": "panda",
        "robot": "Franka Panda",
        "gripper": "Franka default hand",
        "wrist_camera_spheres": "Not modeled",
    },
]

_EMBODIMENT_NAMES = [e["name"] for e in _ROBOT_EMBODIMENTS]


def prompt_robot_type(current_value: str) -> str:
    """Prompt user to select robot embodiment."""
    col_widths = (16, 14, 22, 30)
    header = "{:<{}} {:<{}} {:<{}} {:<{}}".format(
        "Embodiment", col_widths[0],
        "Robot", col_widths[1],
        "Gripper", col_widths[2],
        "Wrist Camera Collision Spheres", col_widths[3],
    )
    separator = " ".join("-" * w for w in col_widths)
    print("Available robot embodiments:")
    print(f"  {header}")
    print(f"  {separator}")
    for e in _ROBOT_EMBODIMENTS:
        row = "{:<{}} {:<{}} {:<{}} {:<{}}".format(
            e["name"], col_widths[0],
            e["robot"], col_widths[1],
            e["gripper"], col_widths[2],
            e["wrist_camera_spheres"], col_widths[3],
        )
        print(f"  {row}")
    options_str = "/".join(_EMBODIMENT_NAMES)
    default = current_value if current_value in _EMBODIMENT_NAMES else None
    while True:
        prompt = f"Robot embodiment ({options_str})" + (f" [{default}]" if default else "") + ": "
        user_input = input(prompt).strip().lower()
        if not user_input and default:
            return default
        if user_input in _EMBODIMENT_NAMES:
            return user_input
        print(f"  ⚠️  Please enter one of: {options_str}.")


def prompt_camera_type(current_value: str) -> str:
    """Prompt user to select camera type (zed or realsense), defaulting to zed."""
    default = current_value if current_value in ("zed", "realsense") else "zed"
    while True:
        user_input = input(f"Camera type (zed/realsense) [{default}]: ").strip().lower()
        if not user_input:
            return default
        if user_input in ("zed", "realsense"):
            return user_input
        print("  ⚠️  Please enter 'zed' or 'realsense'.")


def entrypoint():
    """Main entry point for interactive configuration."""
    script_dir = Path(__file__).parent
    config_path = script_dir.parent / "config" / "tiptop.yml"
    if not config_path.exists():
        print(f"❌ Error: Config file not found at {config_path}")
        return

    print("=" * 60)
    print("🤖 TiPToP Configuration Setup")
    print("=" * 60)
    print("This script will help you configure tiptop.yml.")
    print("Press Enter to keep the current value (when shown).")

    # Load YAML with comment preservation
    yaml = YAML()
    yaml.preserve_quotes = True
    yaml.width = 120
    with open(config_path) as f:
        config = yaml.load(f)

    # Robot configuration
    print("\n🦾 Robot Configuration")
    print("-" * 60)

    config["robot"]["type"] = prompt_robot_type(config["robot"]["type"])

    # Check if host needs updating
    current_host = config["robot"]["host"]
    needs_host = current_host in ["your-ip-address", "localhost", ""]
    print("IP address of workstation running Bamboo controller (NUC in original DROID setup)")
    print("💡 Tip: Run `ifconfig` on that machine to find its IP address")
    config["robot"]["host"] = prompt_with_default(
        "Robot host",
        current_host,
        allow_skip=not needs_host,
    )

    print("💡 Only change ports if you used --control_port or --gripper_port with RunBambooController")
    config["robot"]["port"] = int(prompt_with_default("Robot port", str(config["robot"]["port"])))
    config["robot"]["gripper_port"] = int(prompt_with_default("Gripper port", str(config["robot"]["gripper_port"])))

    # Camera configuration
    print("\n📷 Camera Configuration")
    print("-" * 60)

    # Hand camera
    hand_type = prompt_camera_type(config["cameras"]["hand"]["type"])
    config["cameras"]["hand"]["type"] = hand_type
    serial_tip = "ZED_Explorer" if hand_type == "zed" else "realsense-viewer"
    print(f"💡 Tip: Use {serial_tip} to find the camera's serial number")
    if hand_type == "realsense":
        for key in list(config["cameras"]["hand"].keys()):
            if key not in ("serial", "type"):
                del config["cameras"]["hand"][key]
    current_hand = config["cameras"]["hand"]["serial"]
    needs_hand = current_hand in ["hand-serial", ""]
    config["cameras"]["hand"]["serial"] = prompt_with_default(
        "Hand camera serial number", current_hand, allow_skip=not needs_hand
    )

    # External camera is truly optional
    print("\nExternal camera (optional)")
    print("💡 Useful for recording videos of trials")
    current_external = config["cameras"]["external"]["serial"]
    prev_ext_type = config["cameras"]["external"]["type"]
    ext_type = prompt_camera_type(prev_ext_type)
    config["cameras"]["external"]["type"] = ext_type
    if ext_type != prev_ext_type:
        ext_serial_tip = "ZED_Explorer" if ext_type == "zed" else "realsense-viewer"
        print(f"💡 Tip: Use {ext_serial_tip} to find the camera's serial number")
    if ext_type == "realsense":
        for key in list(config["cameras"]["external"].keys()):
            if key not in ("serial", "type"):
                del config["cameras"]["external"][key]

    needs_external = current_external in ["external-serial", ""]
    if needs_external:
        ext_input = input("External camera serial number (optional, press enter to skip): ").strip()
        if ext_input:
            config["cameras"]["external"]["serial"] = ext_input
    else:
        config["cameras"]["external"]["serial"] = prompt_with_default(
            "External camera serial number (optional)", current_external
        )

    # Perception services
    print("\n🧠 Perception Services")
    print("-" * 60)
    print("💡 Only change if running services on a remote machine")
    config["perception"]["foundation_stereo"]["url"] = prompt_with_default(
        "FoundationStereo URL", config["perception"]["foundation_stereo"]["url"]
    )
    config["perception"]["m2t2"]["url"] = prompt_with_default("M2T2 URL", config["perception"]["m2t2"]["url"])

    # Write back to file
    print(f"\n💾 Saving configuration to: {config_path}")
    with open(config_path, "w") as f:
        yaml.dump(config, f)
    print("✅ Configuration updated successfully!")


if __name__ == "__main__":
    entrypoint()
