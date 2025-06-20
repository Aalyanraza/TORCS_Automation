import msgParser

class CarControl(object):
    '''
    Holds and manages control commands for the TORCS car.
    '''

    def __init__(self, accel=0.0, brake=0.0, gear=1, steer=0.0, clutch=0.0, focus=0, meta=0):
        '''Initialize control parameters'''
        self.parser = msgParser.MsgParser()
        self.actions = {}

        self.accel = accel
        self.brake = brake
        self.gear = gear
        self.steer = steer
        self.clutch = clutch
        self.focus = focus
        self.meta = meta

    def toMsg(self):
        '''Convert control values to message dictionary'''
        self.actions['accel'] = [self.accel]
        self.actions['brake'] = [self.brake]
        self.actions['gear'] = [self.gear]
        self.actions['steer'] = [self.steer]
        self.actions['clutch'] = [self.clutch]
        self.actions['focus'] = [self.focus]
        self.actions['meta'] = [self.meta]

        return self.parser.stringify(self.actions)

    def to_dict(self):
        '''Convert control values to dictionary for logging'''
        return {
            'steer': self.steer,
            'accel': self.accel,
            'brake': self.brake,
            'gear_control': self.gear,  # Renamed to avoid conflict with state.gear
            'clutch': self.clutch,
            'focus': self.focus,
            'meta': self.meta
        }

    # Setters with clamping where applicable
    def setAccel(self, accel):
        self.accel = max(0.0, min(1.0, accel))

    def getAccel(self):
        return self.accel

    def setBrake(self, brake):
        self.brake = max(0.0, min(1.0, brake))

    def getBrake(self):
        return self.brake

    def setGear(self, gear):
        self.gear = max(-1, min(6, gear))

    def getGear(self):
        return self.gear

    def setSteer(self, steer):
        self.steer = max(-1.0, min(1.0, steer))

    def getSteer(self):
        return self.steer

    def setClutch(self, clutch):
        self.clutch = max(0.0, min(1.0, clutch))

    def getClutch(self):
        return self.clutch

    def setFocus(self, focus):
        self.focus = focus

    def getFocus(self):
        return self.focus

    def setMeta(self, meta):
        self.meta = meta

    def getMeta(self):
        return self.meta