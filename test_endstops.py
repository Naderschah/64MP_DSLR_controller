import RPi.GPIO as GPIO
import time
endstops = {'x_min':[21,20], 'x_max':[16,12], 'y_min':[1,7],'y_max':[8,25], 'z_min':[24,23],'z_max':[18,15]}

GPIO.setmode(GPIO.BCM)
def read_pin(pin_nr, check_for=0.05, timeout=2):
    """Reads GPIO pin of pin_nr in BCM numbering
    check_for time in seconds for which signal mustnt fluctuate
    timeout -> if timeout reached without constant signal raises Exception
    """
    # Def last read
    last = GPIO.input(pin_nr)
    start_t = time.time()
    # Check for timeout condition
    while time.time()-start_t < timeout:
        # Check for timing till true condition
        start_c = time.time()
        while time.time()-start_c < check_for:
            new = GPIO.input(pin_nr)
            # break inner while loop if fialed to read consistently
            if new != last: 
                print('Signal varied!')
                break
            # Overwrite last and repeat
            last = new
        # If it reaches here check for condition is satisfied and break for loop
        break
    # Return if true or false
    return last

# Set them up
if endstops is not None:
    for key in endstops:
        # Set one as signal sender
        GPIO.setup(endstops[key][0], GPIO.OUT)
        # Pull high so that signal travels
        GPIO.setup(endstops[key][0], GPIO.HIGH)
        # And other as receiver
        GPIO.setup(endstops[key][1], GPIO.IN, pull_up_down=GPIO.PUD_UP)
        print('Checking endstop for {}'.format(key))
        # Check signal is being received
        if read_pin(endstops[key][1]) == 0:
            # If the above is zero there is no signal through the set up, so raise exception for operator to check if endstop is triggered (and then untrigger) or fix hardware problem
            raise Exception('Endstop does not provide signal, check whats going on\n PIN:{} | NAME: {}'.format(endstops[key][1], key))


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
    return (varied/counter > threshhold)


ls = []
while True:
    ls.append(read_pin(16))
