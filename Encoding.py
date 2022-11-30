import numpy as np
class Encoder:
    def __init__(self):
        pass
    def oneHotEncoding(self,data,size):
        if data >= size:
            error = "Error: Data greater than size\n"
            error += "Data = " + str(data) + ", Size = " + str(size)
            raise ValueError(error)
        hot_encoding = np.zeros(size,dtype=int)
        hot_encoding[data] = 1
        return hot_encoding

    def encodeAccidentSeverity(self,data):
        if data == 1:
            return 1
        if data == 2 or data == 3:
            return data
        return -1
    def encodeAgeBand(self,data):
        return self.oneHotEncoding(data-1,size=10)

    def encodeNumberVehicles(self,data):
        return np.array([data])

    def encodeNumberCasaulties(self,data):
        return np.array([data])

    def encodeDayOfWeek(self,data):
        if 1 <= data <= 7:
            return self.oneHotEncoding(data-1,size=7)
        return -1

    def encodeTime(self,data):
        if 1 <= data <= 4:
            return self.oneHotEncoding(data-1,size=4)
        return -1

    def encodeRoadType(self,data):
        if data == 9:
            return -1
        if (1 <= data <= 3):
            return self.oneHotEncoding(data-1,size=5)
        if data == 6 or data == 7:
            temp_data = data-3
            return self.oneHotEncoding(temp_data,size=5)
        return -1

    def encodeSpeedLimit(self,data):
        if data == 10:
            return self.oneHotEncoding(0,size =8)
        if data == 15:
            return self.oneHotEncoding(1,size= 8)
        temp_data = data // 10
        return self.oneHotEncoding(temp_data, size= 8)

    def encodeJunctionDetail(self,data):
        if data > 9:
            return -1
        if 0<= data < 4:
            return self.oneHotEncoding(data,size=9)
        if  5<= data <= 9:
            return self.oneHotEncoding(data-1,size=9)
        return -1

    def encodeJunctionControl(self,data):
        return self.oneHotEncoding(data,size=5)

    def encodeLightConditions(self,data):
        if data > 1:
            temp_data = data - 3
            return self.oneHotEncoding(temp_data,size=5)
        return self.oneHotEncoding(data-1,size=5)

    def encodeWeatherConditions(self,data):
        if 1 <= data <= 9:
            return self.oneHotEncoding(data-1,size=9)
        return -1

    def encodeRoadSurfaceConditions(self,data):
        if 1<= data <= 5:
            return self.oneHotEncoding(data-1,size=5)
        return -1
    
    def encodeSpecialConditions(self,data):
        if 0<= data <= 7:
            return self.oneHotEncoding(data,size=8)
        return -1
    
    def encodeCarriagewayHazards(self,data):
        if 0 <= data <= 3:
            return self.oneHotEncoding(data,size=6)
        if data == 6 or data == 7:
            temp_data = data-2
            return self.oneHotEncoding(temp_data,size=6)
        return -1

    def encodeUrbanRural(self,data):
        if 1 <= data <= 3:
            return self.oneHotEncoding(data-1, size=3)
        return -1

    def encodePoliceOffier(self,data):
        if 1 <= data <= 3:
            return self.oneHotEncoding(data-1, size=3)
        return -1


    def encodeColumn(self,data, index):
        if data == -1:
            return -1
        
        match index:
            case 0:
                return self.encodeAccidentSeverity(data)
            case 1:
                return self.encodeNumberVehicles(data)
            case 2:
                return self.encodeNumberCasaulties(data)
            case 3:
                return self.encodeDayOfWeek(data)
            case 4:
                return self.encodeTime(data)
            case 5:
                return self.encodeRoadType(data)
            case 6:
                return self.encodeSpeedLimit(data)
            case 7:
                return self.encodeJunctionDetail(data)
            case 8:
                return self.encodeJunctionControl(data)
            case 9:
                return self.encodeLightConditions(data)
            case 10:
                return self.encodeWeatherConditions(data)
            case 11:
                return self.encodeRoadSurfaceConditions(data)
            case 12:
                return self.encodeSpecialConditions(data)
            case 13:
                return self.encodeCarriagewayHazards(data)
            case 14:
                return self.encodeUrbanRural(data)
            case 15:
                return self.encodePoliceOffier(data)

        return np.array([]) 
    #Function receives an array and returns an encoded row
    #Function is in charge of concatenating the outputs from encodeColumn and return the array
    def encodeRow(self, row_input):
        #print(row_input)
        encoded_row = np.array([], dtype= int)
        for i in range(row_input.shape[0]):
            temp_data = self.encodeColumn(row_input[i],i)
            if type(temp_data) != int: #two if statements because you can't compare np arrays with -1
                encoded_row = np.append(encoded_row,temp_data)
            else:
                temp_array = np.array([temp_data])
                encoded_row = np.append(encoded_row,temp_array)

        return encoded_row

