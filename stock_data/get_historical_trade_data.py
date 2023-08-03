import win32com.client
import pandas as pd
import datetime

security = "XFN"

rqm = win32com.client.Dispatch("IressServerApi.RequestManager")

class RowException(Exception):
    pass

def delta_to_time(delta):
    seconds = delta.total_seconds()
    hours = int(seconds // 3600)
    seconds = seconds % 3600
    mins = int(seconds // 60)
    seconds = int(seconds % 60)

    return datetime.time(hour = hours, minute = mins, second = seconds)
    
    


def create_req(date, from_time, to_time):
    request = rqm.CreateMethod("IRESS", "", "PricingTradeHistoricalGet", 0)

    request.Input.Header.Set("WaitForResponse", True, 0)
    request.Input.Header.Set("PageSizeMaximum", 3000)

    request.Input.Parameters.Set("SecurityCode", security, 0)
    request.Input.Parameters.Set("Exchange", "TSX", 0)

    request.Input.Parameters.Set("TradeDate", date)
    request.Input.Parameters.Set("FromTradeTime", from_time)
    request.Input.Parameters.Set("ToTradeTime", to_time)  

    return request


start_day = datetime.date(2018, 1, 1)
open = datetime.time(9, 30, 0)
close = datetime.time(16, 0, 0)


for i in range((datetime.date.today() - start_day).days):
    day = start_day + datetime.timedelta(days=i)

    request = create_req(str(day), str(open), str(close))
    request.Execute()
    
    acc_row_list = []

    while True:
        row_count = request.Output.DataRows.GetCount()
        
        if row_count > 0:

            available_fields = request.Output.DataRows.GetAvailableFields()
            data = request.Output.DataRows.GetRows(available_fields, 0, -1)

            for row in range(row_count): # add data to acc_row_list
                dictrow = {}
                for column in range(len(available_fields)):
                    col_name = available_fields[column]
                    if col_name == "TradePrice" or col_name == "TradeVolume":
                            dictrow[col_name] = data[row][column]
                    elif col_name == "TradeDateTime":       
                            x = data[row][column]
                            dictrow[col_name] = pd.to_datetime(str(x)[:-6])
                acc_row_list.append(dictrow)
            
        if request.PagingState == 1:
            request.Execute()
        else:
            break


    mydf = pd.DataFrame(acc_row_list)
    mydf.to_csv(security + '/' + str(day) + '_' + security + '.csv')
    print(str(day))



