file_path = "./data-01-test-score.csv"

class ReadCSV():
    def __init__(self,file_path):
        self.__file_path = file_path
        self.__data = []
    
    @property
    def data(self):
        return self.__data
    
    
    def read_file(self):
        try:
            f = open(file_path,'r') #열어준 파일을 닫기 위해서 추가
            self.__data = [line.split(',') for line in f.read().split()]
            f.close()
            return self.__data
        except FileNotFoundError as e:
            print(e)
            return []   
            
    def merge_list(self):
        if len(self.__data) == 0:
            self.read_file()
        try:
            return [sum(list(map(
                lambda x: int(x),line))) 
                    for line in self.__data]
        #int(x) 실행시 문자형 string이 있을 경우
        except ValueError:
            print("숫자형이 아닌 데이터가 있습니다.")
            return []

if __name__ == '__main__':
    read_csv = ReadCSV(file_path)

    print(read_csv.read_file())
    print(read_csv.merge_list())