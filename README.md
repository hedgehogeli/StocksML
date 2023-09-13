# StocksML

LSTM (Long Short Term Memory) model taking historical stock data in order to predict percent change in stock/index price between beginning of day to end of day.
Data input is approximately normalized in different ways. 
Result's primary focus is sign correctness, which is then compared against the naive stock prediction's return on interest. 

Data is accessed through IRESS trading software API. Results can only perform similarly to naive ROI due to lack of data variety; API is old limited.

Note that predictions do not simply follow the previous day's results, unlike many LSTM implementations.   


MFC training progression backtest 1:  

![MFC_progression_bt1](https://github.com/hedgehogeli/StocksML/assets/101956761/22ca6294-1221-4781-93d0-694fb62403b7)


MFC normalized %delta test comparison backtest 1:

![MFC_test_cmp_bt1](https://github.com/hedgehogeli/StocksML/assets/101956761/eb29816f-58f7-4bfb-b30a-9d4aef798794)  


MFC training progression backtest 9:

![MFC_progression_bt9](https://github.com/hedgehogeli/StocksML/assets/101956761/08870709-e987-462c-82a9-609b1c5c3575)  


MFC normalized %delta test comparison backtest 9:

![MFC_test_cmp_bt9](https://github.com/hedgehogeli/StocksML/assets/101956761/35f002b3-cf89-4a04-8e71-2c183cda24af)
