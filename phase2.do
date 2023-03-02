* logging
// log using phase2.log, replace

* save "C:\Users\ConquerV\Documents\2022-2023Winter\ECO375\project\data\effr_stock.dta", replace

use "C:\Users\ConquerV\Documents\2022-2023Winter\ECO375\project\data\effr_stock.dta", clear

* Convert effective fund rate from string var to numeric
encode EFFR, generate(effr)

* Set time series
// tsset Date

* Encode effr as a difference
// gen effr_f1 = F1.effr
// gen effr_ma3 = (L1.effr + effr + F1.effr) /3

* Summarize Statistics
summarize effr Daily_Return

* Basic data exploration
scatter Daily_Return effr

* Preliminary Regression Results
reg Daily_Return effr_ma3, robust

log close