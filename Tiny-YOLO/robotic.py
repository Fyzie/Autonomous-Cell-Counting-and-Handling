from PCA9685 import PCA9685
import time
 
pwm = PCA9685(0x40)
pwm.setPWMFreq(50)

def init():
    global step0,step2, step4, step6, step8, step10
    step0 = 1500
    step2 = 1800
    step4 = 650
    step6 = 750
    step8 = 1700
    step10 = 1100
    pwm.setServoPulse(0,step0)
    pwm.setServoPulse(2,step2)
    pwm.setServoPulse(4,step4)
    pwm.setServoPulse(6,step6)
    pwm.setServoPulse(8,step8)
    pwm.setServoPulse(10,step10)
    
def move(channel, step, pos):
    global step0,step2, step4, step6, step8, step10
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
        step2 = pos
    elif channel == 4:
        step4 = pos
    elif channel == 6:
        step6 = pos
    elif channel == 8:
        step8 = pos
    elif channel == 10:
        step10 = pos
    
def control():
    move(10, step10, 1000)
    move(2, step2, 1500)
    move(10, step10, 1600)
    move(2, step2, 1800)
    move(10, step10, 1600)
    move(0, step0, 1200)
    move(2, step2, 1500)
    move(10, step10, 1000)
    move(2, step2, 2000)
    
    #back to origin
    pos0 = 1500
    pos2 = 1800
    pos4 = 650
    pos6 = 750
    pos8 = 1700
    pos10 = 1500
    move(0, step0, pos0)
    move(2, step2, pos2)
    move(4, step4, pos4)
    move(6, step6, pos6)
    move(8, step8, pos8)
    move(10, step10, pos10)

# init()
# control()
