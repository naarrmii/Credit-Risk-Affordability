# Personal study and interview defence notes
*Not for the public repository. This is for you, so you can defend every part of this project under real questioning, including from someone deliberately trying to find a weak spot.*

## How to use this document

Read it properly once, out loud if you can. Do not memorise sentences. Understand the reasoning well enough that you could explain each decision in your own words, in a different order, if the conversation goes somewhere you did not plan for. If a nosy interviewer picks the one thing you skimmed, that is the thing you need to go back and actually understand, not the thing to hope they do not ask about.

## The elevator pitch

Understand the shape of this, then say it in your own words:

> Most credit risk projects only ask whether someone will default. I reframed mine around whether a loan is genuinely affordable for someone's real financial life, which is a more useful question. I built it properly end to end: found and fixed real data problems rather than hiding them, judged the model on the right measures given how rare defaults actually are, made the model explain its own decisions, and tested it directly for fairness, where I found a real problem, argued about the right response, and settled it with evidence rather than opinion. I also built a working, interactive tool, not just a notebook, and I made sure the tool never trusts the statistical model alone, because I found a specific case where trusting it alone would have been a mistake.

## Why this project, personally

You should be able to answer "why does this project matter to you" without reciting the README. Have your own honest version ready. You understand personal finance and are genuinely interested in it, which matters, because faking interest in an interview is obvious. Your accounting background means debt-to-income, credit utilisation and affordability are not just column names to you, they mean something concrete. You considered a project based on your actual placement work, in trade compliance, and deliberately chose not to build on it because you were not passionate about it, which is itself a decision worth being able to explain: a portfolio project you cannot speak about with real interest is a liability, not an asset.

## Technical concepts, explained properly, not just named

You must be able to explain each of these in plain words. If you can only recite the definition, you do not know it well enough yet.

**Class imbalance.** When one outcome you are predicting is much rarer than the other. Here, only 6.7 per cent of roughly 150,000 people actually defaulted. This matters because a model that always guesses "no default" would already be 93.3 per cent accurate while being completely useless, which is exactly why accuracy is the wrong headline measure for this problem.

**ROC-AUC.** A measure of how well a model can rank a genuinely risky case above a genuinely safe one, if you picked one of each at random. A score of 0.5 is a coin flip. A score of 1.0 is perfect separation. The final model scored 0.869.

**PR-AUC.** Similar in spirit but focused specifically on how well the model finds the rare positive class, which matters more here precisely because defaults are rare. A model guessing at random would score close to the base rate, 6.7 per cent, on this measure. The final model scored 0.403, well above that baseline. When you are asked why you reported this alongside ROC-AUC rather than instead of it, the answer is that ROC-AUC alone can look flatteringly high on imbalanced data because it is partly earned against the easy majority class, while PR-AUC does not let that happen.

**Data leakage.** When information that would not actually be available at the moment of a real decision ends up in the training data, making a model look far better than it would ever perform in reality. You were specifically warned that public projects on this kind of dataset often report suspiciously high accuracy, around 98 per cent, because of exactly this. Your own model's realistic, non-suspicious score is itself evidence you avoided it.

**Gradient boosting.** A technique that builds many simple decision rules one after another, where each new one is trained specifically to correct the mistakes of the ones before it, and their combined vote becomes the final prediction. You used scikit-learn's `HistGradientBoostingClassifier` rather than the more commonly namedropped XGBoost, and you should be ready to explain why without sounding defensive: XGBoost was not available in the environment you built this in and you had no way to verify it worked correctly there, so you chose the option you could fully test, which is itself a defensible engineering call, not a downgrade. Functionally they are the same family of technique.

**SHAP.** Short for SHapley Additive exPlanations. A method that works out, for a model's prediction, how much each individual input contributed to the result, in a way that is mathematically fair across all the inputs. It is what turns "the model said high risk" into "the model said high risk mainly because of X and Y." You do not need to be able to derive the underlying game theory to use this well. You do need to be able to say clearly what it is for and what it showed you.

**Recourse.** Going beyond explaining a decision to suggesting what could realistically change it. Your recourse logic deliberately only touches two features, credit utilisation and debt ratio, and explicitly excludes age, dependents, and historical delinquency counts, with the reasons documented directly in the code. Be ready to explain why: those excluded things are either impossible to change or inappropriate to suggest changing, and pretending otherwise would be dishonest to the person reading the advice.

**Fairness, selection rate, false positive rate, false negative rate.** Selection rate is simply what share of a group gets flagged as high risk. False positive rate, among people who would not actually default, what share got wrongly flagged anyway. False negative rate, among people who would actually default, what share got missed. You need all three because a model can look fair on one of these while being badly unfair on another, which is a genuine, well known tension in fairness research, not a flaw in your approach.

**Protected characteristic.** A personal attribute, such as age, that is legally protected against discriminatory treatment under the UK Equality Act. This is why finding age as a top driver of your model's decisions was not something you could treat as a neutral, purely statistical fact.

**Debt-to-income ratio.** Monthly debt payments divided by monthly income, expressed as a percentage. This is a genuine figure real lenders use. You used two long-standing benchmarks: below 36 per cent is generally seen as healthy, and 43 per cent or higher was the actual regulatory cutoff for standard mortgage lending in the United States under the Qualified Mortgage rule, and remains a widely used industry benchmark for a ratio lenders see as risky.

**Leading versus lagging indicator.** A leading indicator changes before trouble becomes visible, such as a sudden drop in income. A lagging indicator only shows up after trouble has been present for a while, such as a missed payment. Your model leans heavily on the lagging indicator, payment history, which is a real, disclosed limitation: it can understate risk for someone whose income has just collapsed but who has not yet missed a payment, simply because that lagging signal has not caught up yet.

**Session state.** In the context of the interactive tool, a way of letting one part of the application remember a value and share it with a different part, across user interactions, rather than each part working in total isolation. You used this specifically so that a decision made in the bank view, where the risk threshold is set, genuinely changes what the consumer view reports, rather than the two views only pretending to be connected.

## Phase by phase, what was decided and why, defensibly

### Phase 1, exploring the data

You loaded roughly 150,000 rows and checked the shape of the problem before touching a model. You found the 93.3 per cent to 6.7 per cent class split, missing income in about 19.8 per cent of rows, missing dependents in about 2.6 per cent, and several suspicious-looking maximum values. If asked why this step matters at all: skipping it is exactly how people end up building on broken assumptions without realising it.

### Phase 2, cleaning the data

Four real issues, each handled with reasoning you can restate:
- One row had age recorded as zero. Dropped, since a real age cannot be inferred and losing one row out of 150,000 costs nothing.
- Revolving credit utilisation had extreme outliers, some in the tens of thousands, when it should realistically sit near 0 to 1. Capped at 2 rather than deleted, because the majority of the "over 1" population sat in a plausible, meaningful 1 to 2 band representing real over-limit distress, and only a small minority were implausible enough to be data errors.
- Missing income and an exploding debt ratio turned out to be connected: rows with missing income had a debt ratio averaging over 1,600, against about 27 where income was present. Missingness was flagged as its own feature before imputing the median income, and debt ratio was capped at 5.
- Three delinquency columns shared an identical, suspicious maximum of 98 on the exact same rows, confirmed by directly checking the overlap. Flagged as a data artefact, then each column capped at its own genuine, non-sentinel maximum.

Also on record here: a real bug where `NumberOfDependents` was flagged as missing in Phase 1 but never actually cleaned, which surfaced later as a crash. This is now one of your best "tell me about a mistake" answers, covered fully below.

### Phase 3, modelling

Logistic regression as a deliberate baseline, then gradient boosting. Logistic regression scored 0.861 ROC-AUC and 0.382 PR-AUC. Gradient boosting scored 0.869 and 0.403, winning on both, and was saved as the final model. At the default threshold, the confusion matrix showed 1,492 of 2,005 real defaulters correctly caught, a 74 per cent recall, at the cost of 5,208 good borrowers wrongly flagged, a precision of only 22 per cent on the positive class. You must be able to say plainly that this is a real trade-off, not a flaw to be embarrassed about, and that `class_weight='balanced'` is precisely why the model leans this way, deliberately trading some precision for a much higher chance of catching real risk.

### Phase 4, explainability and recourse

SHAP confirmed that past delinquency history and credit utilisation dominate the model's decisions by a wide margin, which is legitimate, expected signal, not a spurious pattern. Age ranked third. Both data quality flags from Phase 2 sat near zero importance, a good sanity check that the cleaning work had not leaked noise into the model. The engineered affordability features ranked only moderately on raw predictive power, and you should be ready to explain honestly why they were still worth building: they are not necessarily the model's best predictors, but they are the most interpretable to an actual human being reading the result, which matters for the consumer-facing half of this project specifically.

Recourse was built to re-run the actual model on a hypothetical, changed version of someone's data, rather than adding up SHAP numbers by hand. If asked why: SHAP values are a local approximation around one specific prediction, and for a genuinely different hypothetical scenario on a nonlinear model, actually asking the model again is the more trustworthy approach.

### Phase 5, the fairness investigation

This deserves its own full section below, because it is the strongest material in the whole project.

### Phase 6, the interactive tool

Built as two views sharing one model and one recourse module, so the logic exists in exactly one place rather than being duplicated and risking drifting out of sync. Plotly was used for the charts specifically because the first version, built in matplotlib, was flagged as too plain and non-interactive, and a portfolio piece meant to demonstrate polish should actually demonstrate it.

## The fairness investigation, in full, because this is your strongest material

State the finding precisely, with numbers, every time: eighteen to thirty year olds were flagged as high risk 42.6 per cent of the time, against 8.3 per cent for people over sixty one. False positive rate ranged from 36.2 per cent down to 7.0 per cent across the same groups. False negative rate ran the other way, from 13.2 per cent up to 42.3 per cent. Every group had thousands of people in it, so this is not small sample noise, and the pattern was consistent and monotonic across all four age bands, not just an anomaly in one group.

You initially disagreed with removing age at all, and that disagreement is worth keeping in the story rather than smoothing over: an eighteen year old and a forty two year old with identical current finances are not necessarily equal risks in real life, since life stage carries genuine information, and simply deleting a feature because it looks uncomfortable is not automatically the responsible choice either.

You settled it with a direct test rather than continuing to argue about it. Refitting the model with age completely removed cost only 0.004 ROC-AUC and 0.003 PR-AUC, a negligible drop, which told you age was not carrying much information the other features did not already capture. But the disparity only closed by roughly 22 to 35 per cent, not all the way, meaning most of the effect was travelling through other, correlated features, most likely utilisation and credit history patterns, not age directly.

The final decision: remove age from the model, matching how FICO, the scoring system most real lenders actually use, already works, relying on length of credit history instead of raw age. Age stays in the historical dataset purely to keep monitoring for this gap, but is never collected from anyone using the live consumer tool and cannot be seen by the live model under any circumstance.

If asked to summarise this in one sentence: age is cheap to remove and I would remove it, but removing it did not fix the underlying problem, and I say that plainly rather than claiming a clean win.

## The dashboard build, and every real bug found along the way

This is genuinely good material, because it shows an actual engineering process, not a single clean build. Be ready to walk through any of these without notes.

**The Plotly colour bug.** An eight digit hex colour with a built in transparency value, valid in ordinary web styling, is not accepted by Plotly's own colour handling, which only accepts a plain hex colour or an explicit red, green, blue, transparency format. Fixed by writing a small conversion function. The lesson worth stating out loud: a format being valid in one context does not mean it is valid everywhere, and you cannot assume a library accepts everything a browser does.

**The double-tap slider.** A slider that reads its starting position from a stored value, and then immediately writes back to that same stored value, fights against how the underlying framework is meant to track that widget, causing it to need two interactions before it visually catches up. Fixed by giving the widget a stable identity and letting the framework manage its own memory of it, rather than managing it manually. This is a genuinely well known category of mistake, and knowing the name for it, not just the fix, is worth having ready.

**The vanishing results panel.** A submit button's "was this just clicked" signal is only true for the one screen refresh immediately following the click. Placing an entire results panel, including a secondary interactive input, behind that one-shot signal meant that touching the secondary input made the whole panel think the form had never been submitted, and disappear. Fixed by separately remembering "has this been submitted at all" using persistent memory, rather than relying on "was it just clicked this exact instant."

**The affordability metric that was almost useless.** The first version simply subtracted expenses from income and displayed the pounds left over, which does not actually answer whether a given level of debt is sustainable. Replaced with an actual debt-to-income ratio benchmarked against real figures lenders use, and then made holistic, also factoring in dependents, delinquency history, credit use, and the model's own score, rather than relying on one ratio alone.

**The severity blindness bug.** The affordability verdict counted how many different types of concern were triggered, not how severe any single one was, meaning a debt-to-income ratio of 44 per cent and one of 400 per cent could produce the identical, falsely reassuring verdict if nothing else happened to be flagged. Fixed with an explicit rule: a ratio of 100 per cent or more, meaning the payments alone exceed the person's entire income, now returns an immediate, unambiguous "this is not affordable," regardless of anything else.

**The debt payment figure that quietly went to zero.** An engineered feature representing someone's estimated monthly debt payment was being recalculated from a ratio and their income, rather than using the real, already known figure they had actually typed in. This made sense for the original historical data, which never had a raw payment figure to begin with, only a ratio, but was wrong for a live person, and meant that entering zero income silently zeroed out a real, non-zero debt payment too. Fixed to use the real entered figure directly for a live applicant.

**The misleading age language.** An earlier version of the fairness explanation said age was "still tracked," which reads as if the live tool was quietly collecting age from users, when in fact it referred only to the historical dataset used to test the model. Rewritten to state plainly that the historical test data already contained age before this project existed, that this is the only reason the comparison could be made at all, and that nobody using the live tool is ever asked for it.

**The naive single-model trust gap.** The bank's "flag for review" decision originally checked only the statistical model's own score, completely ignoring the separate affordability check running alongside it. This meant a case where debt payments alone exceeded someone's entire income, but with a clean payment history, could score low risk from the model and never get flagged at all. Fixed so an application is flagged if either signal raises a concern, with the interface stating plainly which one actually caught it. This is your best answer if asked how you would handle a model with a known blind spot: you do not force the model to catch everything, you build an independent second check specifically for what it is weak at, and you disclose which one did the work.

## Known limitations, stated proactively

Naming your own weak points before someone else finds them is a stronger position than being caught out, so use this list on purpose.

- The specific features acting as a stand in for age were never individually identified, only shown to exist through the size of the gap that remained after removing age.
- The model leans on payment history far more than income, meaning it can understate risk for someone whose income has just dropped sharply but who has not yet missed a payment, a genuine and disclosed blind spot, not a hidden one.
- The recourse tool only ever suggests two changes. For someone whose risk is actually driven by historical delinquency, neither suggestion will move their score much, and the tool says so honestly rather than pretending otherwise.
- The dataset is real but over a decade old, and UK consumer credit behaviour, interest rates and typical debt levels have moved on since.
- The fairness testing covers age only. Other protected characteristics were not present in the dataset and so were never tested, which is a gap in coverage, not evidence that no other gap exists.

## A note on how this was built, in case you are asked

Be honest if asked directly. You used AI assistance throughout, including for writing code, catching bugs, and structuring this document. That is now a completely normal part of real engineering work, and no reasonable interviewer holds that against a candidate on its own. What actually matters, and what these notes exist to prove to yourself before anyone else asks, is whether you understand every decision well enough to defend it, extend it, and explain what you would do differently. You caught real problems in this project yourself along the way, including the affordability metric being too simplistic and the age language being misleading, which is itself evidence you understand it, not just accepted it.

## Anticipated interview questions and how to answer them

**"Walk me through this project."**
Use the elevator pitch, then let them steer you into whichever part interests them.

**"Why this dataset?"**
Real, publicly available, personal rather than corporate data, small and simple enough to learn Python on properly, while still being complex enough to support a genuine investigation.

**"Why not just optimise for accuracy?"**
Because only 6.7 per cent of the data actually defaulted, so a model that always guessed "no" would already be 93.3 per cent accurate while being useless. Judged on PR-AUC instead, which does not let that trick inflate the number.

**"Why gradient boosting over XGBoost, since that is more commonly used?"**
Scikit-learn's version was available and fully testable in the environment this was built in, XGBoost was not. Choosing the tool you can actually verify over the more famous name is a defensible engineering decision, not a compromise.

**"Tell me about a mistake you made."**
Two strong, real options. First, missing values in the number of dependents were spotted early but never actually cleaned, which surfaced later as a model crash, fixed at the source rather than patched around the symptom. Second, the very first version of the affordability check was a simple income-minus-expenses figure that turned out to give an obviously wrong "broadly affordable" verdict for a debt-to-income ratio over one hundred per cent, because it counted how many problems were flagged rather than how severe any one of them was, fixed with an explicit severity rule.

**"How did you decide whether to keep or remove age?"**
Tested it rather than arguing about it. Removing it cost almost no accuracy, showing it was not carrying unique signal, but only closed part of the disparity, showing the real issue sat in other, correlated features. Removed it, matching real industry practice, while being honest that this did not fully solve the underlying problem.

**"How would you handle a model with a known blind spot?"**
Do not force the model to catch everything itself. Build an independent second check specifically for the case it is weak at, run both, and disclose plainly which one actually caught a given case, exactly as done here with the model's score and the separate affordability check.

**"What would you do with more time?"**
Identify the specific features acting as a proxy for age, rather than only knowing that some exist. Test the model against more recent UK consumer credit data. Extend the fairness testing to other characteristics if the data allowed it.

**"Did you build this alone?"**
Used AI assistance throughout, as is now normal in real engineering work. What matters is being able to defend every decision in it, which is the entire purpose of this document existing.
