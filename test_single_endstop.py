import RPi.GPIO as GPIO
import time

def read_pin(pin_nr, check_for=0.05, assure=2, threshhold=0.05):
    """Reads GPIO pin of pin_nr in BCM numbering
    check_for time in seconds for which signal mustnt fluctuate

    Pin wont consistently show 0 but does consistently show 1 so if varies for more than check_for it is probably an endstop

    So we do assure time to check it is actually touching
    """
    # var to track how often it varied, and counter for total checks
    varied = 0
    counter = 0
    start_t = time.time()
    while time.time()-start_t < assure:
        start_c = time.time()
        while time.time()-start_c < check_for:
            new = GPIO.input(pin_nr)
            # If its expected
            if new == 1: 
                counter +=1
                break
            # Record variation
            else: 
                varied += 1
                counter +=1
                break
    if (varied/counter > threshhold): print(varied, counter)
    return (varied/counter > threshhold)




pins = [21,20]

GPIO.setmode(GPIO.BCM)

GPIO.setup(pins[0], GPIO.OUT)

GPIO.setup(pins[1], GPIO.IN)

GPIO.setup(pins[0], GPIO.HIGH)

while True:
    print(read_pin(pins[1]))