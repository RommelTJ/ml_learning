# Lesson 5

## 3 cases that are malicious

1. Feedback loops: When your model is controlling the next round of data you get.
2. Systems implemented with no way to identify and address mistakes.
3. Bias in online ad delivery (racism).

## Data collection in genocides

IBM created computers that facilitated the tasks of the Nazis.

## Unintended consequences

Consider how your tech could be used:
* By trolls, harassers
* By authoritarian governments
* For propaganda or misinformation

## Ethics

The discipline dealing with what is good and bag; a set of moral principles.
* Ethics is not the same as religion, law, social norms, or feelings.
* Ethics is not a fixed set of rules.

Ethics is:
1. Well-founded standards of right and wrong that prescribe what humans ought to do.
2. The study and development of one's ethical standards.

Casey Fiesler: Tech Ethics Curricula: A Collection of Syllabi.

## Recourse and Accountability

Common issue: Systems implemented with no way to identify and address mistakes.

Data contains errors:  
* Ex: Auditor found babies in database of California gang members.
* Ex: Credit bureaus found 26% had mistakes in their file.

Technology can be used in unintended ways:
* Facial recognition developed for adults used with children.
* NYPD using facial recognition with pics of Woody Harrelson.
* Abusing confidential databases.

## Feedback Loops and Metrics

Overemphasizing metrics leads to:
* Manipulation
* Gaming
* Myopic focus on short-term goals
* Unexpected negative consequences

Much of AI/ML centers on optimizing a metric.

Examples:  
* England implemented metrics to reduce ER wait times
  * Led to cancelling operations or requiring patients to wait in ambulances.
  * Discrepancies by patients and hospital.
* Essay grading software focuses on sentence length, vocabulary, spelling, subject verb agreement.
  * Can't evaluate things like creativity.
  * Gibberish essays score well.
  * Essays from African-American students receive lower grades from computer vs graders.
  * Essays from students memorizing phrases scored higher.

* Any metric is just a proxy for what you truly care about.  
* Our online environments are designed to be addictive.
* The incentives focus on short term metrics.
* The fundamental business model is around manipulating people's behavior and monopolizing their time.

The major tech platforms (unintentionally) incentivize and promote disinformation:  
* Their design and architecture.
* Their recommendation systems.
* Their business models.

Blitz-scaling Premise: If a company grows big enough and fast enough, profits will eventually follow.  
* Prioritizes speed over efficiency.
* Risks potentially disastrous defeat.
* Investors anoint winners (as opposed to market forces).

How is speed/hypergrowth related to data ethics?
* Hockey-stick growth requires automation and reliance on metrics.
* Prioritizing speed above all else doesn't leave time to reflect on ethics.
* Problems happen at a large scale.

## Getting specific about bias

Different types of bias:
* Historical bias
* Representation bias
* Measurement bias
* Aggregation bias
* Evaluation bias

Representation Bias
* Gender classification worked worse with dark-skinned people.

Benchmark datasets spur on research
* But facial data sets may not be representative.
* ImageNet images are from the West. Very few from outside the Western World.

Recidivism algorithm used in prison sentencing
* False positive rate for recidivism was 44% for African Americans. 

Historical Bias
* Even given a perfect sampling and feature selection, a structural issue might exist that leads to historical bias.
* Might be mitigated by talking to domain experts and those impacted.

Measurement Bias
* Ex: Trying to predict strokes, they look at metrics to predict strokes.
* Many of those metrics are from high utility patients vs low utility patients.
* People that utilize healthcare a lot will go to the doctor for Sinusitis and Stroke.

There are many studies on racial bias.

Why does algorithmic bias matter?
* Machine Learning can amplify bias.
* Algorithms are used differently than human decision makers.
  * People assume algorithms are error free.
  * More likely to be implemented without an appeals process.
  * Often used at scala.
  * Are Often cheap.

Humans are biased, so why does algorithmic bias matter?
* Machine learning can create feedback loops.
* Machine learning can amplify bias.
* Algorithms and humans are used differently.
* Technology is power. And with that comes responsibility.

## Questions to ask about AI

* Should we even be doing this?
  * Engineers tend to respond to problems with building stuff. Sometimes the answer is not to build.
  * Ex: Racial or sexual recognition.
* What bias is in the data?
  * Be aware of the context of your data.
* Can the code and data be audited?
  * Companies creating technology that can impact people.
* What are error rates for different groups?
  * Dark-skinned error rates for facial recognition.
* What is the accuracy of a simple rule-based alternative?
  * Have a baseline.
* What processes are in place to handle appeals or mistakes?
  * Need to have a process for recourse.
* How diverse is the team that built it?

Fairness, Accountability and Transparency are important, but they aren't everything.
* Ex: Being transparent about how to turn old people into mulch.

## Disinformation

* Ex: Russian trolls organized both sides of a protest in Texas.
* Disinformation includes orchestrated campaigns of manipulation. 
* Disinformation is an ecosystem. 
  * Claire Wardle: The Trumpet of Amplification. 
* Examples:
  * /r/SubSimulatorGPT2 
  * Katie Jones on Twitter 
  * Unicorn story essay 
  * Thispersondoesnotexist.com 
* Online discussion may be swamped with fake, manipulative agents. 
* Disinformation is a cybersecurity problem.
