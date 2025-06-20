class MsgParser(object):
    def __init__(self):
        '''Constructor'''
        pass
    
    def parse(self, str_sensors):
        '''Return a dictionary with tags and values from the UDP message'''
        sensors = {}
        
        if not str_sensors or '(' not in str_sensors:
            print("Invalid sensor string:", str_sensors)
            return None
        
        b_open = str_sensors.find('(')
        while b_open >= 0:
            b_close = str_sensors.find(')', b_open)
            if b_close < 0:
                print("Unmatched parenthesis in:", str_sensors)
                return None
            substr = str_sensors[b_open + 1: b_close]
            items = substr.split()
            if len(items) < 2:
                print("Problem parsing substring:", substr)
            else:
                try:
                    # Convert all values to floats if possible
                    value = [float(x) for x in items[1:]]
                    sensors[items[0]] = value if len(value) > 1 else value[0]
                except ValueError:
                    print(f"Non-numeric value in substring: {substr}. Storing as string.")
                    sensors[items[0]] = items[1:]
            b_open = str_sensors.find('(', b_close)
        
        return sensors
    
    def stringify(self, dictionary):
        '''Build an UDP message from a dictionary'''
        msg = ''
        
        for key, value in dictionary.items():
            if value is not None and (isinstance(value, list) and len(value) > 0 and value[0] is not None):
                msg += '(' + key
                for val in value:
                    msg += ' ' + str(val)
                msg += ')'
            elif value is not None:
                # In case value is a scalar, treat it as a single item list
                msg += '(' + key + ' ' + str(value) + ')'
        
        return msg