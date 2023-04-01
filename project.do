use "C:\Users\ConquerV\Documents\2022-2023Winter\ECO475\project\data\fed_spy_data.dta", clear  

* Log file
// log using 'effr.csv', replace

* Basic summary statistics
summarize FFF_Price FFF_Change M1_ImpliedRate WeeklyReturn Close

* generate datetime variable from strings
gen date = date(Week, "YMD")
// gen Chg% = double(_Chg)
format date %td

* set timeseries data and plot
tsset date, week
// gen Ret_F1= L1.WeeklyReturn
// xtline WeeklyReturn M1_ImpliedRate FFF_Change
// tsline WeeklyReturn
// tsline FFF_Change

// dfuller WeeklyReturn
// dfuller FFF_Change

// vecrank WeeklyReturn FFF_Change, lags(1)
// vecrank WeeklyReturn FFF_Change, trend(constant) max ic

// xtcointtest kao WeeklyReturn FFF_Change

* Run Baseline Regression
reg WeeklyReturn FFF_Change M1_ImpliedRate, robust

* save files
* save()