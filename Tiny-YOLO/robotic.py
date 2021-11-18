from PCA9685 import PCA9685
import time
 
pwm = PCA9685(0x40)
pwm.setPWMFreq(50)

def init():
    global step0,step2, step4, step6
    step0 = 1500
    step2 = 1800
    step4 = 600
    step6 = 750
    pwm.setServoPulse(0,step0)
    pwm.setServoPulse(2,step2)
    pwm.setServoPulse(4,step4)
    pwm.setServoPulse(6,step6)

def move(channel, step, pos):
    global step0,step2, step4, step6
    if step < pos:
        for i in range(step,pos,10):
            pwm.setServoPulse(channel,i)
            time.sleep(0.05)
    elif step > pos:
        for i in range(step,pos,-10):
            pwm.setServoPulse(channel,i)
            time.sleep(0.05)
    if channel == 0:
        step0 = pos
    elif channel == 2:
        step2 = pos;
    elif channel == 4:
        step4 = pos
    elif channel == 6:
        step6 = pos;
    
def control():
    move(0, step0, 2000)
    move(2, step2, 2000)
    move(2, step2, 1800)

