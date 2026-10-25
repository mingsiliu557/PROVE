# VisualSimpleQA Failure Analysis V2

## Metric Caveat

- VisualSimpleQA step0 has 20 rows with missing `exact_acc`; 19 of these are marked `correct=True/reward=1` even though most predictions are `unknown`.
- Therefore the experiment-level reward/correct metric says VisualSimpleQA is flat, but strict normalized exact says it improves modestly. The improvement is still much smaller than OCRBench and A-OKVQA.

## Overall Comparison

| Dataset | n | official/correct step0 -> final | strict exact step0 -> final | reward step0 -> final | unknown step0 -> final | changed prediction |
|---|---:|---:|---:|---:|---:|---:|
| VisualSimpleQA | 500 | 9.20% -> 9.20% | 5.40% -> 9.20% | 9.20% -> 9.20% | 83.0% -> 65.6% | 21.6% |
| OCRBench v1 | 1000 | 14.70% -> 52.00% | 14.70% -> 52.00% | 14.70% -> 52.00% | 73.6% -> 29.0% | 47.1% |
| A-OKVQA DA val | 1075 | 20.47% -> 56.00% | 13.86% -> 36.56% | 18.14% -> 48.78% | 65.6% -> 10.2% | 61.3% |

## Training Rollout Comparison

| Dataset | HQ unique/repeat | density peak correct | majority correct | correct present | avg valid/32 | avg unique valid | avg correct count | corr density/freq adv |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| VisualSimpleQA | 16/4 | 67.5% | 77.5% | 95.0% | 27.9 | 9.5 | 14.9 | 0.950 |
| OCRBench v1 | 20/0 | 85.0% | 90.0% | 95.0% | 16.6 | 2.9 | 13.2 | 0.975 |
| A-OKVQA DA val | 20/0 | 47.5% | 65.0% | 72.5% | 28.5 | 13.6 | 6.9 | 1.000 |

## VisualSimpleQA Transition Matrices

Strict normalized exact:
- wrong -> wrong: 451
- wrong -> correct: 22
- correct -> correct: 24
- correct -> wrong: 3

Official reward/correct, including step0 missing-exact anomaly:
- wrong -> wrong: 432
- wrong -> correct: 22
- correct -> correct: 24
- correct -> wrong: 22

## VisualSimpleQA Category Breakdown, Strict Exact

| Category | n | step0 | final | delta | wrong->right | right->wrong | unknown step0 -> final |
|---|---:|---:|---:|---:|---:|---:|---:|
| research and education | 111 | 1.8% | 4.5% | +2.7 | 3 | 0 | 91.0% -> 78.4% |
| company and brand | 92 | 3.3% | 6.5% | +3.3 | 3 | 0 | 83.7% -> 63.0% |
| other | 70 | 17.1% | 22.9% | +5.7 | 6 | 2 | 51.4% -> 32.9% |
| history | 54 | 3.7% | 7.4% | +3.7 | 3 | 1 | 92.6% -> 77.8% |
| art | 45 | 2.2% | 4.4% | +2.2 | 1 | 0 | 93.3% -> 75.6% |
| geography | 41 | 14.6% | 19.5% | +4.9 | 2 | 0 | 58.5% -> 31.7% |
| film and television entertainment | 28 | 0.0% | 3.6% | +3.6 | 1 | 0 | 100.0% -> 85.7% |
| sports | 28 | 0.0% | 10.7% | +10.7 | 3 | 0 | 100.0% -> 67.9% |
| games | 17 | 5.9% | 5.9% | +0.0 | 0 | 0 | 88.2% -> 82.4% |
| politics | 14 | 0.0% | 0.0% | +0.0 | 0 | 0 | 100.0% -> 100.0% |

## Strict wrong -> correct examples

- `visualsimpleqa_da-test-1` | category=research and education | gt=`yes`
  - Q: [image] Can humans improve digestion by eating the fruits of this crop? Answer with a short direct answer.
  - step0: pred=`yes humans can improve digestion by eating fruits of this crop` raw=`|Yes, humans can improve digestion by eating the fruits of this crop.<|im_end|>` reward=0.0 strict=0
  - final: pred=`yes` raw=` Yes<|im_end|>` reward=1.0 strict=1
- `visualsimpleqa_da-test-20` | category=company and brand | gt=`1916`
  - Q: [image] In which year is the car brand established? Answer with a short direct answer.
  - step0: pred=`unknown` raw=`<|im_end|>` reward=0.0 strict=0
  - final: pred=`1916` raw=`?1916<|im_end|>` reward=1.0 strict=1
- `visualsimpleqa_da-test-23` | category=research and education | gt=`vitamin c`
  - Q: [image] Which vitamin is most abundant in the fruit shown in the image? Answer with a short direct answer.
  - step0: pred=`unknown` raw=`}<|im_end|>` reward=0.0 strict=0
  - final: pred=`vitamin c` raw=`} Vitamin C<|im_end|>` reward=1.0 strict=1
- `visualsimpleqa_da-test-68` | category=sports | gt=`china`
  - Q: [image] From which country does this sport originate? Answer with a short direct answer.
  - step0: pred=`unknown` raw=`<|im_end|>` reward=0.0 strict=0
  - final: pred=`china` raw=`] China<|im_end|>` reward=1.0 strict=1
- `visualsimpleqa_da-test-78` | category=company and brand | gt=`yes`
  - Q: [image] Can we have fast food here? Answer with a short direct answer.
  - step0: pred=`unknown` raw=`<|im_end|>` reward=0.0 strict=0
  - final: pred=`yes` raw=` Yes<|im_end|>` reward=1.0 strict=1
- `visualsimpleqa_da-test-123` | category=other | gt=`thanksgiving`
  - Q: [image] What holiday would you consider this to be? Answer with a short direct answer.
  - step0: pred=`unknown` raw=`<|im_end|>` reward=0.0 strict=0
  - final: pred=`thanksgiving` raw=`} Thanksgiving<|im_end|>` reward=1.0 strict=1
- `visualsimpleqa_da-test-124` | category=history | gt=`1769`
  - Q: [image] In which year was the person represented by the sculpture born? Answer with a short direct answer.
  - step0: pred=`unknown` raw=`<|im_end|>` reward=0.0 strict=0
  - final: pred=`1769` raw=`} 1769<|im_end|>` reward=1.0 strict=1
- `visualsimpleqa_da-test-133` | category=research and education | gt=`first quarter`
  - Q: [image] From the perspective of an observer in the Northern Hemisphere, what phase of the celestial body is depicted in the image? Answer with a short direct answer.
  - step0: pred=`unknown` raw=`<|im_end|>` reward=0.0 strict=0
  - final: pred=`first quarter` raw=`} First quarter<|im_end|>` reward=1.0 strict=1
- `visualsimpleqa_da-test-140` | category=art | gt=`united kingdom`
  - Q: [image] Which country is the author of the famous novel shown on this bus from? Answer with a short direct answer.
  - step0: pred=`bulgaria` raw=` Bulgaria<|im_end|>` reward=0.0 strict=0
  - final: pred=`united kingdom` raw=`} United Kingdom<|im_end|>` reward=1.0 strict=1
- `visualsimpleqa_da-test-151` | category=other | gt=`no`
  - Q: [image] Does the prototype animal of this doll in the picture have teeth? Answer with a short direct answer.
  - step0: pred=`unknown` raw=`}<|im_end|>` reward=0.0 strict=0
  - final: pred=`no` raw=`] No<|im_end|>` reward=1.0 strict=1
- `visualsimpleqa_da-test-156` | category=other | gt=`scared`
  - Q: [image] Is the cat more likely to feel happy or scared? Answer with a short direct answer.
  - step0: pred=`unknown` raw=`<|im_end|>` reward=0.0 strict=0
  - final: pred=`scared` raw=`] scared<|im_end|>` reward=1.0 strict=1
- `visualsimpleqa_da-test-164` | category=other | gt=`halloween`
  - Q: [image] Which festival are they preparing for? Answer with a short direct answer.
  - step0: pred=`unknown` raw=`<|im_end|>` reward=0.0 strict=0
  - final: pred=`halloween` raw=`} Halloween<|im_end|>` reward=1.0 strict=1

## Strict correct -> wrong examples

- `visualsimpleqa_da-test-111` | category=other | gt=`plastic`
  - Q: [image] What is the most commonly used material that the blue item in the person's hand is made of? Answer with a short direct answer.
  - step0: pred=`plastic` raw=`} plastic<|im_end|>` reward=1.0 strict=1
  - final: pred=`polyurethane` raw=`} Polyurethane<|im_end|>` reward=0.0 strict=0
- `visualsimpleqa_da-test-422` | category=other | gt=`no`
  - Q: [image] Is it true that this building was not originally white? Answer with a short direct answer.
  - step0: pred=`no` raw=`} No<|im_end|>` reward=1.0 strict=1
  - final: pred=`no it is not true that this building was not originally white` raw=`) No, it is not true that this building was not originally white.<|im_end|>` reward=0.0 strict=0
- `visualsimpleqa_da-test-455` | category=history | gt=`yes`
  - Q: [image] Is this a traditional game from ancient China? Answer with a short direct answer.
  - step0: pred=`yes` raw=` Yes<|im_end|>` reward=1.0 strict=1
  - final: pred=`no` raw=` No<|im_end|>` reward=0.0 strict=0

## Still wrong examples

- `visualsimpleqa_da-test-0` | category=company and brand | gt=`2007`
  - Q: [image] In which year did the steakhouse named after this breed of cattle become part of the Darden Restaurant family? Answer with a short direct answer.
  - step0: pred=`unknown` raw=`<|im_end|>` reward=0.0 strict=0
  - final: pred=`unknown` raw=`<|im_end|>` reward=0.0 strict=0
- `visualsimpleqa_da-test-2` | category=company and brand | gt=`toothpastes and oral care`
  - Q: [image] What is the content in the bottle with blue and red colors most likely used for? Answer with a short direct answer.
  - step0: pred=`toothpaste` raw=`} toothpaste<|im_end|>` reward=0.0 strict=0
  - final: pred=`toothpaste` raw=`} toothpaste<|im_end|>` reward=0.0 strict=0
- `visualsimpleqa_da-test-3` | category=company and brand | gt=`wolfsburg germany`
  - Q: [image] In which city is the headquarters of the brand that produces the blue-and-white car located? Answer with a short direct answer.
  - step0: pred=`unknown` raw=`<|im_end|>` reward=0.0 strict=0
  - final: pred=`unknown` raw=`<|im_end|>` reward=0.0 strict=0
- `visualsimpleqa_da-test-4` | category=film and television entertainment | gt=`pyotr ilyich tchaikovsky`
  - Q: [image] Who composed the classic ballet closely associated with the content shown in the picture? Answer with a short direct answer.
  - step0: pred=`unknown` raw=`<|im_end|>` reward=0.0 strict=0
  - final: pred=`unknown` raw=`<|im_end|>` reward=0.0 strict=0
- `visualsimpleqa_da-test-5` | category=company and brand | gt=`vf corporation`
  - Q: [image] Which company owns the brand of the backpack? Answer with a short direct answer.
  - step0: pred=`unknown` raw=`<|im_end|>` reward=0.0 strict=0
  - final: pred=`jansport` raw=`} JanSport<|im_end|>` reward=0.0 strict=0
- `visualsimpleqa_da-test-6` | category=geography | gt=`jingdezhen`
  - Q: [image] Which city in China is famous for producing this type of handicraft? Answer with a short direct answer.
  - step0: pred=`unknown` raw=`<|im_end|>` reward=0.0 strict=0
  - final: pred=`jiujiang` raw=`|Jiujiang<|im_end|>` reward=0.0 strict=0
- `visualsimpleqa_da-test-7` | category=history | gt=`1973`
  - Q: [image] In which year did the opera house open? Answer with a short direct answer.
  - step0: pred=`unknown` raw=`<|im_end|>` reward=0.0 strict=0
  - final: pred=`unknown` raw=`<|im_end|>` reward=0.0 strict=0
- `visualsimpleqa_da-test-8` | category=sports | gt=`nanjing 2014 youth olympic games`
  - Q: [image] In which sports meeting did this sport make its debut on the Olympic stage? Answer with a short direct answer.
  - step0: pred=`unknown` raw=`<|im_end|>` reward=0.0 strict=0
  - final: pred=`unknown` raw=`<|im_end|>` reward=0.0 strict=0
- `visualsimpleqa_da-test-9` | category=geography | gt=`italy`
  - Q: [image] In which country did this vegetable originate? Answer with a short direct answer.
  - step0: pred=`china` raw=`} China<|im_end|>` reward=0.0 strict=0
  - final: pred=`china` raw=`} China<|im_end|>` reward=0.0 strict=0
- `visualsimpleqa_da-test-10` | category=history | gt=`19th century`
  - Q: [image] In which century was this kind of musical instrument invented? Answer with a short direct answer.
  - step0: pred=`unknown` raw=`<|im_end|>` reward=0.0 strict=0
  - final: pred=`18th century` raw=`} 18th century<|im_end|>` reward=0.0 strict=0
- `visualsimpleqa_da-test-11` | category=history | gt=`19th century`
  - Q: [image] In which century did this type of banana first make its way to America via Toronto? Answer with a short direct answer.
  - step0: pred=`unknown` raw=`<|im_end|>` reward=0.0 strict=0
  - final: pred=`unknown` raw=`<|im_end|>` reward=0.0 strict=0
- `visualsimpleqa_da-test-14` | category=other | gt=`yes`
  - Q: [image] Is this bird a migratory bird? Answer with a short direct answer.
  - step0: pred=`it is not possible to determine from image whether bird is migratory bird or not` raw=` It is not possible to determine from the image whether the bird is a migratory bird or not.<|im_end` reward=0.0 strict=0
  - final: pred=`it is not possible to determine from image whether bird is migratory bird or not` raw=` It is not possible to determine from the image whether the bird is a migratory bird or not.<|im_end` reward=0.0 strict=0
- `visualsimpleqa_da-test-15` | category=research and education | gt=`google`
  - Q: [image] Which company developed the large language model that shares its name with this tree? Answer with a short direct answer.
  - step0: pred=`unknown` raw=`<|im_end|>` reward=0.0 strict=0
  - final: pred=`unknown` raw=`<|im_end|>` reward=0.0 strict=0
- `visualsimpleqa_da-test-16` | category=film and television entertainment | gt=`1961`
  - Q: [image] In which year was the Disney movie about this pet first released? Answer with a short direct answer.
  - step0: pred=`unknown` raw=`<|im_end|>` reward=0.0 strict=0
  - final: pred=`unknown` raw=`<|im_end|>` reward=0.0 strict=0

## Step0 missing exact / false positive examples

- `visualsimpleqa_da-test-240` gt=`arthur b lamb` pred=`unknown` reward=0.0 correct=False category=research and education
- `visualsimpleqa_da-test-241` gt=`joghem van loghem` pred=`unknown` reward=1.0 correct=True category=research and education
- `visualsimpleqa_da-test-242` gt=`smokey joe williams` pred=`unknown` reward=1.0 correct=True category=sports
- `visualsimpleqa_da-test-243` gt=`september 3` pred=`unknown` reward=1.0 correct=True category=other
- `visualsimpleqa_da-test-244` gt=`march 7 1966 february 7 1970` pred=`unknown` reward=1.0 correct=True category=research and education
- `visualsimpleqa_da-test-245` gt=`l der rutenberg heinrich beck and f g thomas may` pred=`unknown` reward=1.0 correct=True category=company and brand
- `visualsimpleqa_da-test-246` gt=`hainan` pred=`unknown` reward=1.0 correct=True category=politics
- `visualsimpleqa_da-test-247` gt=`november 25 1952` pred=`unknown` reward=1.0 correct=True category=company and brand
- `visualsimpleqa_da-test-248` gt=`april 10 1885` pred=`unknown` reward=1.0 correct=True category=research and education
- `visualsimpleqa_da-test-249` gt=`november 8 2023` pred=`unknown` reward=1.0 correct=True category=games
- `visualsimpleqa_da-test-250` gt=`1964 1965 new york world s fair` pred=`unknown` reward=1.0 correct=True category=history
- `visualsimpleqa_da-test-251` gt=`alexa dembek` pred=`unknown` reward=1.0 correct=True category=research and education

## Wrong density pseudo-label examples on VisualSimpleQA train

- `visualsimpleqa_da_20_hq_major_correct-train-13` | gt=`halloween` | density_peak=`off` | majority=`unknown` | correct_count=0 | valid_ratio=0.25 | category=other
  - Q: [image] What holiday are the decorations on the left table most associated with? Answer with a short direct answer.
  - counter: `unknown`:24, `reapers`:1, `based on image decorations on left table suggest association with thanksgiving`:1, `turkeys`:1, `off`:1, `semiclassical`:1, `themost associated with decorations on left table is thanksgiving`:1, `b`:1
- `visualsimpleqa_da_20_hq_major_correct-train-5` | gt=`coca cola` | density_peak=`pepsi` | majority=`unknown` | correct_count=0 | valid_ratio=0.28 | category=company and brand
  - Q: [image] Which cola company uses this animal as its mascot? Answer with a short direct answer.
  - counter: `unknown`:23, `pepsi`:4, `polar bear`:1, `organism`:1, `net energy`:1, `soviet union pepsi`:1, `movetohead`:1
- `visualsimpleqa_da_20_hq_major_correct-train-9` | gt=`united kingdom` | density_peak=`germany` | majority=`united kingdom` | correct_count=6 | valid_ratio=0.91 | category=art
  - Q: [image] Which country is the author of the famous novel shown on this bus from? Answer with a short direct answer.
  - counter: `united kingdom`:6, `bulgaria`:5, `unknown`:3, `reland`:3, `uk`:2, `poland`:2, `implicit ur`:1, `luxembourg`:1
- `visualsimpleqa_da_20_hq_major_correct-train-4` | gt=`30` | density_peak=`32` | majority=`30` | correct_count=6 | valid_ratio=0.97 | category=other
  - Q: [image] How many permanent teeth does this animal typically have in adulthood? Answer with a short direct answer.
  - counter: `30`:6, `32`:4, `18`:2, `fifteen`:1, `three`:1, `52`:1, `38 permanent teeth`:1, `starting age of 2 cat typically has 42 permanent teeth`:1
- `visualsimpleqa_da_20_hq_major_correct-train-14` | gt=`japan` | density_peak=`american` | majority=`unknown` | correct_count=4 | valid_ratio=0.69 | category=other
  - Q: [image] Which country's cuisine is this chef likely cooking? Answer with a short direct answer.
  - counter: `unknown`:10, `japanese`:4, `japan`:4, `armenia`:2, `american`:2, `based on image it s difficult to determine exact country but chef seems to be cooking traditional cuisine that may be associated with east asian southeast asian american mexican or potentially japanese given style of cooking equipment and utensils`:1, `based on image it is not possible to determine chef s nationality or specific country s cuisine from which they are cooking`:1, `country whose culture is brazilian`:1
- `visualsimpleqa_da_20_hq_major_correct-train-12` | gt=`china` | density_peak=`japan` | majority=`china` | correct_count=8 | valid_ratio=1.00 | category=geography
  - Q: [image] Which country has the most of this animal? Answer with a short direct answer.
  - counter: `china`:8, `poland`:6, `armenia`:3, `bulgaria`:2, `exert`:1, `thailand`:1, `russia`:1, `britain`:1
- `visualsimpleqa_da_20_hq_major_correct-train-8` | gt=`no` | density_peak=`yes it is prohibited` | majority=`no` | correct_count=17 | valid_ratio=1.00 | category=other
  - Q: [image] In the scene shown in the image, is a vehicle permitted to pass through the arched gate? Answer with a short direct answer.
  - counter: `no`:17, `no vehicle is not permitted to pass through arched gate`:7, `no it is not permitted to pass through gate`:1, `vehicle is not permitted to pass through arched gate`:1, `no because prohibitory road sign on right shows not rentable`:1, `no vehicle is not permitted to pass through`:1, `no vehicles are not permitted to pass through arched gate as indicated by road sign`:1, `sign indicates that vehicles are not allowed to pass through gate`:1
- `visualsimpleqa_da_20_hq_major_correct-train-12` | gt=`china` | density_peak=`mongolia` | majority=`china` | correct_count=7 | valid_ratio=1.00 | category=geography
  - Q: [image] Which country has the most of this animal? Answer with a short direct answer.
  - counter: `china`:7, `poland`:4, `armenia`:4, `bulgaria`:3, `mongolia`:1, `ukraine`:1, `country with most pandas is china`:1, `estonia japan china`:1

## Interpretation

1. VisualSimpleQA does get a format/invalid reduction: unknown drops from 83.0% to 65.6%, but many new non-unknown answers are plausible wrong world-knowledge guesses.
2. Under strict exact, VisualSimpleQA improves only 5.4% -> 9.2% (+3.8), far below OCRBench (+37.3) and A-OKVQA (+35.5). Under the experiment reward/correct field it appears flat because the step0 file contains 19 false-positive missing-exact rows.
3. OCRBench is text extraction: once the model stops emitting blanks and shortens answers, many predictions become exactly correct. A-OKVQA has broad but short object/action answers and the soft direct-answer reward accepts common synonyms.
4. VisualSimpleQA is much more open-world. Many questions require image recognition plus external facts: brand owner, founding year, country/city, historical date, author/composer. A 20-sample self-training signal cannot supply those facts.
5. The density reward is nearly frequency reward on these rollouts. `corr density/freq adv` is 0.95 for VisualSimpleQA, and wrong dense clusters like `pepsi` for `coca cola`, `32` for `30`, or `germany` for `united kingdom` show the mechanism can amplify confident wrong answers.