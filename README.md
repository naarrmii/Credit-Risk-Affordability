# Can You Actually Afford This Loan?
### A credit risk project built around one idea: a lending decision should be explainable and fair, not just accurate.

## Why I built this

I have a BSc in Accounting and Finance, and I am partway through an MSc in Data Science, currently on a placement year doing trade compliance analysis. I wanted a portfolio project that showed more than the ability to follow a tutorial. Most public projects on this kind of dataset stop at "clean the data, fit a model, report an accuracy number." I wanted to build something that actually behaved like a real financial decision, with all the messiness, trade-offs and second-guessing that involves, and that I could explain properly to anyone, technical or not.

The idea I kept coming back to was affordability. Most credit models only ask one question: will this person default? That is a useful question for a lender, but it is not the same question a real person is actually asking, which is closer to: can I actually manage this? I wanted to build something that took that second question seriously, not just the first one.

## What this project actually does

In short, it is a tool that estimates the risk of someone falling seriously behind on loan repayments, based on their financial history, and explains that estimate in plain terms rather than as an unexplained number. It goes further than that in three ways that most similar projects do not:

- It explains **why** it reached a particular score, using a method that attributes the result to specific pieces of information, rather than leaving the model as a black box.
- It suggests **what could realistically change** the outcome, but only for things a real person can actually act on, not things about their past or their identity that cannot be changed.
- It was tested directly for **whether it treats people fairly**, found a real problem, and the finished project honestly reports what was found and what was and was not fixed, rather than only showing the flattering numbers.

The finished result is a small web application with two views: one for a bank's risk team, showing the technical detail, and one for an ordinary person, letting them type in their own numbers and get an estimate along with practical next steps.

## Why this data

Within consumer credit data, I picked a dataset called [Give Me Some Credit](https://www.kaggle.com/c/GiveMeSomeCredit), published on [Kaggle](https://www.kaggle.com/c/GiveMeSomeCredit), a data science website. I chose it deliberately over two well known alternatives. Home Credit Default Risk spreads its information across several linked tables, which suits a much broader data engineering exercise, but would have pulled focus away from the actual question I wanted to explore. LendingClub has around 145 columns, adding considerable width without adding much depth to that same question. Give Me Some Credit has one single table, ten meaningful columns, and around one hundred and fifty thousand real, anonymised borrowers, which let me stay focused on the substance of the project: the cleaning, the feature engineering and the fairness testing, rather than on wrangling spread out, loosely related data. It is also specific to individual people rather than companies, which fits the personal finance angle I wanted from the start.

## What is actually in the data

Each row is one anonymised person who applied for credit at some point in the past. The columns describe things like their income, how much of their available credit they are currently using, how many credit accounts they have open, how many times they have paid late in the past, and how many people depend on them financially. The single most important column is the target, a yes or no answer to whether that person went 90 days or more late on a payment within the following two years. That target is what the whole model is trained to predict.

One important, slightly surprising fact about this data shaped a lot of the project: only about 6.7 per cent of the roughly one hundred and fifty thousand people in the dataset actually had this happen to them. In data science, this is called class imbalance, meaning the two outcomes you are trying to tell apart are not evenly represented. It matters because a lazy model could simply guess "no" for everybody and still be right 93 per cent of the time while being completely useless. Almost every later decision in this project, from how the model was judged to how it was trained, exists because of that one fact.

## Cleaning the data honestly

Before touching any model, the data was checked carefully for problems, because building on bad data quietly produces a broken result. Four real issues were found and are worth explaining plainly, because how they were handled says more about the quality of this project than any single accuracy number does.

**A single impossible age.**
- One row listed someone's age as zero.
- There is no way to guess a real age for that row, so it was removed.
- Losing one row out of one hundred and fifty thousand changes nothing.

**Credit use above one hundred per cent.**
- One column records how much of a person's available credit they are using, which should sit between zero and one hundred per cent.
- A small number of rows recorded values in the thousands, which is not possible and is almost certainly a data entry fault.
- Deleting these rows would have thrown away real information from people who were genuinely, if only moderately, over their limit.
- Every value was capped at a sensible upper bound instead, keeping that real signal while removing the nonsensical extremes.

**A debt figure tangled up with missing income.**
- Around one in five people in the dataset had no income recorded at all.
- Wherever income was missing, a separate column meant to represent debt relative to income was also wildly, impossibly large, in some cases in the hundreds of thousands.
- This showed the two problems were connected, most likely because a calculation upstream had broken when it tried to divide by a missing number.
- The fix: flag which rows had missing income as their own piece of information, fill the gap with a sensible average, then cap the debt figure at a reasonable ceiling.

**A hidden placeholder pretending to be real data.**
- Three separate columns, each recording a different band of late payment, shared the exact same suspiciously specific maximum value of 98.
- Three independent real behaviours landing on the same unusual number is not a coincidence. It is a sign of a shared computer system code standing in for "unknown," left in the data by mistake.
- Confirmed by checking whether the same rows were affected across all three columns, which they were.
- Each column was capped at its own highest genuine value, and a new flag recorded which rows had been affected, so the model could learn from the fact that something was off without being misled by the fake number itself.

## Teaching the model what affordability means

The dataset does not contain a ready made "can this person afford their life" figure, so three new figures were built from what was available:

- An estimate of how much someone is likely paying towards debts each month, worked out from their debt ratio and income.
- Disposable income: that figure subtracted from income, a rough estimate of what is left over each month.
- Total past delinquencies: all of someone's past late payments added into a single, more readable total.

These figures are not the model's biggest statistical drivers, explained honestly further down, but they are what makes an otherwise abstract score mean something to an actual person.

## Building the model

Two models were built and compared. The first, logistic regression, is one of the simplest and most established methods in statistics for telling two outcomes apart, and it acts as a baseline, a number to try and beat. The second, gradient boosting, is a more modern, more flexible technique that builds a large number of very simple decision rules one after another, each one correcting the mistakes of the ones before it, and combines them into a single strong prediction. It can pick up on patterns and combinations that a simpler model cannot.

Both were judged on two figures, not the one most tutorials reach for first. Plain accuracy, meaning the percentage of correct guesses, is a poor judge of a model like this, precisely because of the 6.7 per cent imbalance already mentioned. Instead, both models were judged on a measure called ROC-AUC, roughly how well a model can rank a genuinely risky person above a genuinely safe one, and a second, more telling measure called PR-AUC, which focuses specifically on how well the model finds the rare group that actually goes on to default. Gradient boosting won on both and was kept as the final model, scoring realistically, not suspiciously perfectly, which is itself a good sign that nothing had leaked from the future into the training data by mistake.

## Making the model explain itself

A model that only outputs a number is not good enough for a decision this important. A method called SHAP was used to work out, for the model as a whole and for any single prediction, how much each individual piece of information pushed the result up or down. This confirmed something reassuring: the model leans most heavily on a person's past payment history and how much of their credit they are using, which is exactly the kind of sensible, expected signal a credit model should rely on, not a random or spurious pattern.

## Testing whether the model is fair

This is the part of the project I am most proud of, because it did not go the way I expected, and I changed my approach based on evidence rather than assumption.

While examining what drove the model's decisions, age turned out to be the third most important factor. Age is also specifically protected under the Equality Act in the United Kingdom, so rather than assume that was fine because the overall numbers looked reasonable, it was tested directly. The finding was stark. Eighteen to thirty year olds were being flagged as high risk roughly five times as often as people over sixty one, a gap far too large and far too consistent across thousands of people to be a coincidence.

My first instinct was that this was not necessarily wrong. An eighteen year old and a forty two year old with identical current finances are not necessarily equal risks in real life, since life stage carries genuine information. I pushed back on simply removing age for this reason. The way to settle a disagreement like this properly is with evidence, not opinion, so the model was refitted with age completely removed from training. The result: accuracy barely moved, a difference so small it suggested the model did not actually need age once it already had someone's payment history and credit use. However, removing age only closed around a quarter to a third of the original gap, not all of it, meaning most of the unfairness was travelling through other, related pieces of information rather than age itself.

The final decision was to remove age from the model entirely, which matches how [FICO](https://www.myfico.com/credit-education/whats-in-your-credit-score), the credit scoring system used by most real lenders, already works, relying on the length of someone's credit history rather than their age directly. Age is still kept in the historical data used to test the model, purely to keep checking for this gap going forward, but nobody using the finished tool is ever asked for their age, and the live model cannot see it under any circumstance. This was not presented as a solved problem. The remaining gap is disclosed honestly as further work, not hidden behind a single clean headline figure.

## Turning a score into advice

A model that only says "high risk" is not very useful to the person being scored. This is where recourse comes in: working out what would actually help, but deliberately only suggesting two changes, how much of someone's credit they are using, and their debt relative to their income. Nothing else is ever suggested, on purpose. A person's age, their number of dependents, and their past payment history cannot be changed by suggestion, so telling someone to alter them would be both pointless and inappropriate. This distinction, between things that can genuinely be acted on and things that cannot, is written directly into the code as a documented, deliberate design decision.

## Would a bank actually trust this alone

No, and that is an important, deliberate part of the design, not an oversight. The model on its own can be shown a case where someone's monthly debt payments genuinely exceed their entire income, yet still return a low risk score, simply because their payment history happens to look clean so far. That is not the model malfunctioning. It reflects a real, disclosed limitation: this model leans heavily on payment history, which only shows up after someone has struggled for a while, and comparatively lightly on income, which can change overnight. Relying on the statistical model alone for a case like that would be genuinely careless.

For that reason, no decision here rests on the statistical model by itself. A second, separate, rule based affordability check runs alongside it, using well established figures real lenders actually use, such as the ratio of someone's debts to their income. If either the model or the affordability check raises a serious concern, the application is flagged, and the interface says plainly which of the two caught it. This mirrors how real regulated lending already works, where an affordability assessment is legally required alongside, not instead of, credit scoring.

## Building something people can actually use

What came out of this is not just a set of analysis notebooks, but a small, interactive web application with two views. The bank facing view shows the model's real performance figures, what drives its decisions, and the full fairness investigation, including a slider that lets you see, in real time, the trade off between catching genuine risk and wrongly flagging good borrowers. The consumer facing view lets an ordinary person type in their own numbers and get a plain English estimate, a breakdown of what mattered most for their specific result, and a small calculator to check whether taking on additional credit looks sustainable before they commit to it.

## What I would still like to explore

Being honest about what is left undone is part of doing this properly. The exact features acting as a stand in for age have not been individually identified, only shown to exist. A genuine fix would involve testing each candidate feature in turn to see which ones are carrying the disparity. The dataset itself, while real, is over a decade old and UK specific consumer credit behaviour has moved on since. And the model's own blind spot around sudden income loss, discussed above, is disclosed rather than solved, since properly addressing it would likely need additional, more current data the original dataset does not contain.

## Tech stack

- **[Python](https://www.python.org):** the programming language used throughout
- **[pandas](https://pandas.pydata.org) and [NumPy](https://numpy.org):** cleaning and working with the data
- **[scikit-learn](https://scikit-learn.org):** logistic regression and gradient boosting (`HistGradientBoostingClassifier`), train and test splitting, and evaluation
- **[SHAP](https://shap.readthedocs.io):** explaining individual model predictions
- **[Fairlearn](https://fairlearn.org):** cross-checking the fairness metrics computed by hand
- **[Matplotlib](https://matplotlib.org):** charts inside the analysis notebooks
- **[Plotly](https://plotly.com/python/):** the interactive charts inside the finished dashboard
- **[Streamlit](https://docs.streamlit.io):** the two-view interactive web application
- **[Jupyter Notebook](https://jupyter.org):** the step-by-step analysis, Phases 1 through 5
- **[joblib](https://joblib.readthedocs.io):** saving and loading the trained model between notebooks and the dashboard
- **[Git](https://git-scm.com) and [GitHub](https://github.com):** version control and hosting this project

## How to reproduce this

```bash
git clone <this-repo-url>
cd credit-risk-affordability
pip install -r requirements.txt
```
Download `cs-training.csv` from the [Kaggle Give Me Some Credit competition page](https://www.kaggle.com/c/GiveMeSomeCredit) and place it in the `data` folder, then run the notebooks in the `notebooks` folder in order, from 01 through to 05. Once those have run, start the interactive application from the project's root folder with:
```bash
streamlit run app.py
```

## About me

Built by Imraan Morolong. MSc Data Science, BSc Accounting and Finance. This project combines an accounting background with data science methods to explore a real, regulator relevant problem in consumer lending.
