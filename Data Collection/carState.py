import msgParser

class CarState(object):
    def __init__(self, msg=None):
        '''Constructor'''
        self.parser = msgParser.MsgParser()
        self.sensors = None
        self.angle = None
        self.curLapTime = None
        self.damage = None
        self.distFromStart = None
        self.distRaced = None
        self.focus = None
        self.fuel = None
        self.gear = None
        self.lastLapTime = None
        self.opponents = None
        self.racePos = None
        self.rpm = None
        self.speedX = None
        self.speedY = None
        self.speedZ = None
        self.track = None
        self.trackPos = None
        self.wheelSpinVel = None
        self.z = None
        if msg:
            self.setFromMsg(msg)
    
    def setFromMsg(self, str_sensors):
        self.sensors = self.parser.parse(str_sensors)
        if self.sensors is None:
            return
        
        self.setAngleD()
        self.setCurLapTimeD()
        self.setDamageD()
        self.setDistFromStartD()
        self.setDistRacedD()
        self.setFocusD()
        self.setFuelD()
        self.setGearD()
        self.setLastLapTimeD()
        self.setOpponentsD()
        self.setRacePosD()
        self.setRpmD()
        self.setSpeedXD()
        self.setSpeedYD()
        self.setSpeedZD()
        self.setTrackD()
        self.setTrackPosD()
        self.setWheelSpinVelD()
        self.setZD()
    
    def toMsg(self):
        self.sensors = {}
        
        self.sensors['angle'] = [self.angle]
        self.sensors['curLapTime'] = [self.curLapTime]
        self.sensors['damage'] = [self.damage]
        self.sensors['distFromStart'] = [self.distFromStart]
        self.sensors['distRaced'] = [self.distRaced]
        self.sensors['focus'] = self.focus
        self.sensors['fuel'] = [self.fuel]
        self.sensors['gear'] = [self.gear]
        self.sensors['lastLapTime'] = [self.lastLapTime]
        self.sensors['opponents'] = self.opponents
        self.sensors['racePos'] = [self.racePos]
        self.sensors['rpm'] = [self.rpm]
        self.sensors['speedX'] = [self.speedX]
        self.sensors['speedY'] = [self.speedY]
        self.sensors['speedZ'] = [self.speedZ]
        self.sensors['track'] = self.track
        self.sensors['trackPos'] = [self.trackPos]
        self.sensors['wheelSpinVel'] = self.wheelSpinVel
        self.sensors['z'] = [self.z]
        
        return self.parser.stringify(self.sensors)
    
    def to_dict(self):
        # Flatten all sensor data for CSV logging
        result = {
            'timestamp': '',
            'episode': 0,
            'step': 0,
            'angle': self.angle or 0.0,
            'curLapTime': self.curLapTime or 0.0,
            'damage': self.damage or 0.0,
            'distFromStart': self.distFromStart or 0.0,
            'distRaced': self.distRaced or 0.0,
            'fuel': self.fuel or 0.0,
            'gear': self.gear or 0,
            'lastLapTime': self.lastLapTime or 0.0,
            'racePos': self.racePos or 0,
            'rpm': self.rpm or 0.0,
            'speedX': self.speedX or 0.0,
            'speedY': self.speedY or 0.0,
            'speedZ': self.speedZ or 0.0,
            'trackPos': self.trackPos or 0.0,
            'z': self.z or 0.0
        }
        # Flatten track (19 sensors)
        for i in range(19):
            result[f'track_{i}'] = self.track[i] if self.track and i < len(self.track) else 0.0
        # Flatten opponents (36 sensors)
        for i in range(36):
            result[f'opponents_{i}'] = self.opponents[i] if self.opponents and i < len(self.opponents) else 0.0
        # Flatten wheelSpinVel (4 sensors)
        for i in range(4):
            result[f'wheelSpinVel_{i}'] = self.wheelSpinVel[i] if self.wheelSpinVel and i < len(self.wheelSpinVel) else 0.0
        return result

    def getFloatD(self, name):
        try:
            val = self.sensors[name]
        except KeyError:
            val = None
        
        if val is not None:
            if isinstance(val, list):
                val = float(val[0]) if val else 0.0
            else:
                val = float(val)
        else:
            val = 0.0
        
        return val
    
    def getFloatListD(self, name):
        try:
            val = self.sensors[name]
        except KeyError:
            val = None
        
        if val is not None:
            if not isinstance(val, list):
                val = [float(val)]
            else:
                val = [float(v) for v in val]
        
        return val
    
    def getIntD(self, name):
        try:
            val = self.sensors[name]
        except KeyError:
            val = None
        
        if val is not None:
            if isinstance(val, list):
                val = int(val[0]) if val else 0
            else:
                val = int(val)
        
        return val
    
    @staticmethod
    def get_field_names():
        return [
            'timestamp', 'episode', 'step',
            'angle', 'curLapTime', 'damage', 'distFromStart', 'distRaced', 'fuel', 'gear',
            'lastLapTime', 'racePos', 'rpm', 'speedX', 'speedY', 'speedZ', 'trackPos', 'z',
            *[f'track_{i}' for i in range(19)],
            *[f'opponents_{i}' for i in range(36)],
            *[f'wheelSpinVel_{i}' for i in range(4)],
            'steer', 'accel', 'brake', 'gear_control', 'clutch', 'focus', 'meta'
        ]
    
    def setAngle(self, angle):
        self.angle = angle
    
    def setAngleD(self):        
        self.angle = self.getFloatD('angle')
        
    def getAngle(self):
        return self.angle
    
    def setCurLapTime(self, curLapTime):
        self.curLapTime = curLapTime
    
    def setCurLapTimeD(self):
        self.curLapTime = self.getFloatD('curLapTime')
    
    def getCurLapTime(self):
        return self.curLapTime
    
    def setDamage(self, damage):
        self.damage = damage
    
    def setDamageD(self):
        self.damage = self.getFloatD('damage')
        
    def getDamage(self):
        return self.damage
    
    def setDistFromStart(self, distFromStart):
        self.distFromStart = distFromStart
    
    def setDistFromStartD(self):
        self.distFromStart = self.getFloatD('distFromStart')
    
    def getDistFromStart(self):
        return self.distFromStart
    
    def setDistRaced(self, distRaced):
        self.distRaced = distRaced
    
    def setDistRacedD(self):
        self.distRaced = self.getFloatD('distRaced')
    
    def getDistRaced(self):
        return self.distRaced
    
    def setFocus(self, focus):
        self.focus = focus
    
    def setFocusD(self):
        self.focus = self.getFloatListD('focus')
    
    def setFuel(self, fuel):
        self.fuel = fuel
    
    def setFuelD(self):
        self.fuel = self.getFloatD('fuel')
    
    def getFuel(self):
        return self.fuel
    
    def setGear(self, gear):
        self.gear = gear
    
    def setGearD(self):
        self.gear = self.getIntD('gear')
    
    def getGear(self):
        return self.gear
    
    def setLastLapTime(self, lastLapTime):
        self.lastLapTime = lastLapTime
    
    def setLastLapTimeD(self):
        self.lastLapTime = self.getFloatD('lastLapTime')
    
    def setOpponents(self, opponents):
        self.opponents = opponents
        
    def setOpponentsD(self):
        self.opponents = self.getFloatListD('opponents')
    
    def getOpponents(self):
        return self.opponents
    
    def setRacePos(self, racePos):
        self.racePos = racePos
    
    def setRacePosD(self):
        self.racePos = self.getIntD('racePos')
    
    def getRacePos(self):
        return self.racePos
    
    def setRpm(self, rpm):
        self.rpm = rpm
    
    def setRpmD(self):
        self.rpm = self.getFloatD('rpm')
    
    def getRpm(self):
        return self.rpm
    
    def setSpeedX(self, speedX):
        self.speedX = speedX
    
    def setSpeedXD(self):
        self.speedX = self.getFloatD('speedX')
    
    def getSpeedX(self):
        return self.speedX
    
    def setSpeedY(self, speedY):
        self.speedY = speedY
    
    def setSpeedYD(self):
        self.speedY = self.getFloatD('speedY')
    
    def getSpeedY(self):
        return self.speedY
    
    def setSpeedZ(self, speedZ):
        self.speedZ = speedZ
    
    def setSpeedZD(self):
        self.speedZ = self.getFloatD('speedZ')
    
    def getSpeedZ(self):
        return self.speedZ
    
    def setTrack(self, track):
        self.track = track
    
    def setTrackD(self):
        self.track = self.getFloatListD('track')
    
    def getTrack(self):
        return self.track
    
    def setTrackPos(self, trackPos):
        self.trackPos = trackPos
    
    def setTrackPosD(self):
        self.trackPos = self.getFloatD('trackPos')
    
    def getTrackPos(self):
        return self.trackPos
    
    def setWheelSpinVel(self, wheelSpinVel):
        self.wheelSpinVel = wheelSpinVel
    
    def setWheelSpinVelD(self):
        self.wheelSpinVel = self.getFloatListD('wheelSpinVel')
    
    def getWheelSpinVel(self):
        return self.wheelSpinVel
    
    def setZ(self, z):
        self.z = z
    
    def setZD(self):
        self.z = self.getFloatD('z')
    
    def getZ(self):
        return self.z