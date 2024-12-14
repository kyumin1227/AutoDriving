import os

def enable_pwm(pin, addr1, value1, addr2, value2):
    """
    Enable a PWM pin by writing to specific memory addresses.

    Args:
        pin (int): Pin number (for logging purposes).
        addr1 (str): First memory address to write.
        value1 (str): Value to write to the first address.
        addr2 (str): Second memory address to write.
        value2 (str): Value to write to the second address.
    """
    try:
        print(f"Enabling PWM on Pin {pin}...")
        # Execute the first memory write
        os.system(f"sudo busybox devmem {addr1} 32 {value1}")
        print(f"Configured {addr1} with value {value1}.")
        
        # Execute the second memory write
        os.system(f"sudo busybox devmem {addr2} 32 {value2}")
        print(f"Configured {addr2} with value {value2}.")
        
        print(f"PWM on Pin {pin} enabled successfully.")
    except Exception as e:
        print(f"Error enabling PWM on Pin {pin}: {e}")

# Enable Pin 32 / PWM0
enable_pwm(pin=32, addr1="0x700031fc", value1="0x45", addr2="0x6000d504", value2="0x2")

# Enable Pin 33 / PWM2
enable_pwm(pin=33, addr1="0x70003248", value1="0x46", addr2="0x6000d100", value2="0x00")
