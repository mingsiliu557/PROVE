# VisualSimpleQA Direct-Answer Failure Analysis

## Overall Metrics

| Dataset | n | step0 exact | final exact | delta | step0 reward | final reward | unknown step0 -> final | changed pred |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| VisualSimpleQA | 500 | 9.20% | 9.20% | +0.00 | 9.20% | 9.20% | 83.0% -> 65.6% | 21.6% |
| OCRBench v1 | 1000 | 14.70% | 52.00% | +37.30 | 14.70% | 52.00% | 73.6% -> 29.0% | 47.1% |
| A-OKVQA DA val | 1075 | 20.47% | 56.00% | +35.53 | 18.14% | 48.78% | 65.6% -> 10.2% | 61.3% |

## Training Rollout / Pseudo-Label Quality

| Dataset | rollout groups | HQ unique/repeat | correct present | density peak correct | majority correct | avg valid / 32 | avg unique valid answers | avg top valid count | avg correct count | corr(density,freq adv) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| VisualSimpleQA | 40 | 16/4 | 95.0% | 67.5% | 77.5% | 27.9 | 9.5 | 16.3 | 14.9 | 0.950 |
| OCRBench v1 | 40 | 20/0 | 95.0% | 85.0% | 90.0% | 16.6 | 2.9 | 13.7 | 13.2 | 0.975 |
| A-OKVQA DA val | 40 | 20/0 | 72.5% | 47.5% | 65.0% | 28.5 | 13.6 | 11.8 | 6.9 | 1.000 |

## VisualSimpleQA Transition Matrix

- `wrong_to_wrong`: 432
- `wrong_to_correct`: 22
- `correct_to_correct`: 24
- `correct_to_wrong`: 22

## VisualSimpleQA Category Breakdown

| Category | n | step0 | final | delta | wrong->right | right->wrong | unknown step0 -> final |
|---|---:|---:|---:|---:|---:|---:|---:|
| research and education | 111 | 7.2% | 4.5% | -2.7 | 3 | 6 | 91.0% -> 78.4% |
| company and brand | 92 | 5.4% | 6.5% | +1.1 | 3 | 2 | 83.7% -> 63.0% |
| other | 70 | 20.0% | 22.9% | +2.9 | 6 | 4 | 51.4% -> 32.9% |
| history | 54 | 7.4% | 7.4% | +0.0 | 3 | 3 | 92.6% -> 77.8% |
| art | 45 | 4.4% | 4.4% | +0.0 | 1 | 1 | 93.3% -> 75.6% |
| geography | 41 | 14.6% | 19.5% | +4.9 | 2 | 0 | 58.5% -> 31.7% |
| film and television entertainment | 28 | 7.1% | 3.6% | -3.6 | 1 | 2 | 100.0% -> 85.7% |
| sports | 28 | 3.6% | 10.7% | +7.1 | 3 | 1 | 100.0% -> 67.9% |
| games | 17 | 17.6% | 5.9% | -11.8 | 0 | 2 | 88.2% -> 82.4% |
| politics | 14 | 7.1% | 0.0% | -7.1 | 0 | 1 | 100.0% -> 100.0% |

## VisualSimpleQA Top Predictions

Step0 top predictions: `unknown`:415, `yes`:14, `armenia`:6, `poland`:5, `no`:4, `1`:3, `london`:2, `bangkok`:2, `australia`:2, `yes humans can improve digestion by eating fruits of this crop`:1

Final top predictions: `unknown`:328, `yes`:19, `armenia`:6, `1`:6, `no`:5, `china`:3, `london`:3, `poland`:3, `france`:3, `new york`:3

## VisualSimpleQA wrong -> correct examples

- `visualsimpleqa_da-test-1` | category=research and education | gt=`yes`
  - Q: [image] Can humans improve digestion by eating the fruits of this crop? Answer with a short direct answer.
  - step0: pred=`yes humans can improve digestion by eating fruits of this crop` raw=`|Yes, humans can improve digestion by eating the fruits of this crop.<|im_end|>` reward=0.000 exact=0
  - final: pred=`yes` raw=` Yes<|im_end|>` reward=1.000 exact=1
- `visualsimpleqa_da-test-20` | category=company and brand | gt=`1916`
  - Q: [image] In which year is the car brand established? Answer with a short direct answer.
  - step0: pred=`unknown` raw=`<|im_end|>` reward=0.000 exact=0
  - final: pred=`1916` raw=`?1916<|im_end|>` reward=1.000 exact=1
- `visualsimpleqa_da-test-23` | category=research and education | gt=`vitamin c`
  - Q: [image] Which vitamin is most abundant in the fruit shown in the image? Answer with a short direct answer.
  - step0: pred=`unknown` raw=`}<|im_end|>` reward=0.000 exact=0
  - final: pred=`vitamin c` raw=`} Vitamin C<|im_end|>` reward=1.000 exact=1
- `visualsimpleqa_da-test-68` | category=sports | gt=`china`
  - Q: [image] From which country does this sport originate? Answer with a short direct answer.
  - step0: pred=`unknown` raw=`<|im_end|>` reward=0.000 exact=0
  - final: pred=`china` raw=`] China<|im_end|>` reward=1.000 exact=1
- `visualsimpleqa_da-test-78` | category=company and brand | gt=`yes`
  - Q: [image] Can we have fast food here? Answer with a short direct answer.
  - step0: pred=`unknown` raw=`<|im_end|>` reward=0.000 exact=0
  - final: pred=`yes` raw=` Yes<|im_end|>` reward=1.000 exact=1
- `visualsimpleqa_da-test-123` | category=other | gt=`thanksgiving`
  - Q: [image] What holiday would you consider this to be? Answer with a short direct answer.
  - step0: pred=`unknown` raw=`<|im_end|>` reward=0.000 exact=0
  - final: pred=`thanksgiving` raw=`} Thanksgiving<|im_end|>` reward=1.000 exact=1
- `visualsimpleqa_da-test-124` | category=history | gt=`1769`
  - Q: [image] In which year was the person represented by the sculpture born? Answer with a short direct answer.
  - step0: pred=`unknown` raw=`<|im_end|>` reward=0.000 exact=0
  - final: pred=`1769` raw=`} 1769<|im_end|>` reward=1.000 exact=1
- `visualsimpleqa_da-test-133` | category=research and education | gt=`first quarter`
  - Q: [image] From the perspective of an observer in the Northern Hemisphere, what phase of the celestial body is depicted in the image? Answer with a short direct answer.
  - step0: pred=`unknown` raw=`<|im_end|>` reward=0.000 exact=0
  - final: pred=`first quarter` raw=`} First quarter<|im_end|>` reward=1.000 exact=1
- `visualsimpleqa_da-test-140` | category=art | gt=`united kingdom`
  - Q: [image] Which country is the author of the famous novel shown on this bus from? Answer with a short direct answer.
  - step0: pred=`bulgaria` raw=` Bulgaria<|im_end|>` reward=0.000 exact=0
  - final: pred=`united kingdom` raw=`} United Kingdom<|im_end|>` reward=1.000 exact=1
- `visualsimpleqa_da-test-151` | category=other | gt=`no`
  - Q: [image] Does the prototype animal of this doll in the picture have teeth? Answer with a short direct answer.
  - step0: pred=`unknown` raw=`}<|im_end|>` reward=0.000 exact=0
  - final: pred=`no` raw=`] No<|im_end|>` reward=1.000 exact=1
- `visualsimpleqa_da-test-156` | category=other | gt=`scared`
  - Q: [image] Is the cat more likely to feel happy or scared? Answer with a short direct answer.
  - step0: pred=`unknown` raw=`<|im_end|>` reward=0.000 exact=0
  - final: pred=`scared` raw=`] scared<|im_end|>` reward=1.000 exact=1
- `visualsimpleqa_da-test-164` | category=other | gt=`halloween`
  - Q: [image] Which festival are they preparing for? Answer with a short direct answer.
  - step0: pred=`unknown` raw=`<|im_end|>` reward=0.000 exact=0
  - final: pred=`halloween` raw=`} Halloween<|im_end|>` reward=1.000 exact=1

## VisualSimpleQA correct -> wrong examples

- `visualsimpleqa_da-test-111` | category=other | gt=`plastic`
  - Q: [image] What is the most commonly used material that the blue item in the person's hand is made of? Answer with a short direct answer.
  - step0: pred=`plastic` raw=`} plastic<|im_end|>` reward=1.000 exact=1
  - final: pred=`polyurethane` raw=`} Polyurethane<|im_end|>` reward=0.000 exact=0
- `visualsimpleqa_da-test-241` | category=research and education | gt=`joghem van loghem`
  - Q: [image] Who received the honorary doctorate from the university in 1986? Answer with a short direct answer.
  - step0: pred=`unknown` raw=`<|im_end|>` reward=1.000 exact=1
  - final: pred=`unknown` raw=`<|im_end|>` reward=0.000 exact=0
- `visualsimpleqa_da-test-242` | category=sports | gt=`smokey joe williams`
  - Q: [image] Which player who ever played in this league was inducted into the National Baseball Hall of Fame in 1999? Answer with a short direct answer.
  - step0: pred=`unknown` raw=`<|im_end|>` reward=1.000 exact=1
  - final: pred=`unknown` raw=`<|im_end|>` reward=0.000 exact=0
- `visualsimpleqa_da-test-243` | category=other | gt=`september 3`
  - Q: [image] On which day and month is the National Day celebrated in the country where this place is located? Answer with a short direct answer.
  - step0: pred=`unknown` raw=`<|im_end|>` reward=1.000 exact=1
  - final: pred=`unknown` raw=`<|im_end|>` reward=0.000 exact=0
- `visualsimpleqa_da-test-244` | category=research and education | gt=`march 7 1966 february 7 1970`
  - Q: [image] What were the start and end dates (year, month, and day) of the tenure of the 8th president of the university represented by the emblem in the photo? Answer with a short direct answer.
  - step0: pred=`unknown` raw=`<|im_end|>` reward=1.000 exact=1
  - final: pred=`unknown` raw=`<|im_end|>` reward=0.000 exact=0
- `visualsimpleqa_da-test-245` | category=company and brand | gt=`l der rutenberg heinrich beck and f g thomas may`
  - Q: [image] Who founded the brand advertised on the umbrellas? Answer with a short direct answer.
  - step0: pred=`unknown` raw=`<|im_end|>` reward=1.000 exact=1
  - final: pred=`unknown` raw=`<|im_end|>` reward=0.000 exact=0
- `visualsimpleqa_da-test-246` | category=politics | gt=`hainan`
  - Q: [image] Which province was the second stop on this president's visit to China in 2018? Answer with a short direct answer.
  - step0: pred=`unknown` raw=`<|im_end|>` reward=1.000 exact=1
  - final: pred=`unknown` raw=`<|im_end|>` reward=0.000 exact=0
- `visualsimpleqa_da-test-247` | category=company and brand | gt=`november 25 1952`
  - Q: [image] What day, month, and year did the newspaper headquartered in this building begin publication in Osaka? Answer with a short direct answer.
  - step0: pred=`unknown` raw=`<|im_end|>` reward=1.000 exact=1
  - final: pred=`unknown` raw=`<|im_end|>` reward=0.000 exact=0
- `visualsimpleqa_da-test-248` | category=research and education | gt=`april 10 1885`
  - Q: [image] On which day, month, and year was the university where the woman completed her undergraduate studies founded? Answer with a short direct answer.
  - step0: pred=`unknown` raw=`<|im_end|>` reward=1.000 exact=1
  - final: pred=`unknown` raw=`<|im_end|>` reward=0.000 exact=0
- `visualsimpleqa_da-test-249` | category=games | gt=`november 8 2023`
  - Q: [image] On which day, month, and year did the first aircraft associated with this game start flying? Answer with a short direct answer.
  - step0: pred=`unknown` raw=`<|im_end|>` reward=1.000 exact=1
  - final: pred=`unknown` raw=`<|im_end|>` reward=0.000 exact=0
- `visualsimpleqa_da-test-250` | category=history | gt=`1964 1965 new york world s fair`
  - Q: [image] What was the entertainment project in the building originally created for? Answer with a short direct answer.
  - step0: pred=`unknown` raw=`<|im_end|>` reward=1.000 exact=1
  - final: pred=`unknown` raw=`<|im_end|>` reward=0.000 exact=0
- `visualsimpleqa_da-test-251` | category=research and education | gt=`alexa dembek`
  - Q: [image] Who received the Innovation Leadership Award from the organization indicated in the photo in 2023? Answer with a short direct answer.
  - step0: pred=`unknown` raw=`<|im_end|>` reward=1.000 exact=1
  - final: pred=`unknown` raw=`<|im_end|>` reward=0.000 exact=0

## VisualSimpleQA still wrong examples

- `visualsimpleqa_da-test-0` | category=company and brand | gt=`2007`
  - Q: [image] In which year did the steakhouse named after this breed of cattle become part of the Darden Restaurant family? Answer with a short direct answer.
  - step0: pred=`unknown` raw=`<|im_end|>` reward=0.000 exact=0
  - final: pred=`unknown` raw=`<|im_end|>` reward=0.000 exact=0
- `visualsimpleqa_da-test-2` | category=company and brand | gt=`toothpastes and oral care`
  - Q: [image] What is the content in the bottle with blue and red colors most likely used for? Answer with a short direct answer.
  - step0: pred=`toothpaste` raw=`} toothpaste<|im_end|>` reward=0.000 exact=0
  - final: pred=`toothpaste` raw=`} toothpaste<|im_end|>` reward=0.000 exact=0
- `visualsimpleqa_da-test-3` | category=company and brand | gt=`wolfsburg germany`
  - Q: [image] In which city is the headquarters of the brand that produces the blue-and-white car located? Answer with a short direct answer.
  - step0: pred=`unknown` raw=`<|im_end|>` reward=0.000 exact=0
  - final: pred=`unknown` raw=`<|im_end|>` reward=0.000 exact=0
- `visualsimpleqa_da-test-4` | category=film and television entertainment | gt=`pyotr ilyich tchaikovsky`
  - Q: [image] Who composed the classic ballet closely associated with the content shown in the picture? Answer with a short direct answer.
  - step0: pred=`unknown` raw=`<|im_end|>` reward=0.000 exact=0
  - final: pred=`unknown` raw=`<|im_end|>` reward=0.000 exact=0
- `visualsimpleqa_da-test-5` | category=company and brand | gt=`vf corporation`
  - Q: [image] Which company owns the brand of the backpack? Answer with a short direct answer.
  - step0: pred=`unknown` raw=`<|im_end|>` reward=0.000 exact=0
  - final: pred=`jansport` raw=`} JanSport<|im_end|>` reward=0.000 exact=0
- `visualsimpleqa_da-test-6` | category=geography | gt=`jingdezhen`
  - Q: [image] Which city in China is famous for producing this type of handicraft? Answer with a short direct answer.
  - step0: pred=`unknown` raw=`<|im_end|>` reward=0.000 exact=0
  - final: pred=`jiujiang` raw=`|Jiujiang<|im_end|>` reward=0.000 exact=0
- `visualsimpleqa_da-test-7` | category=history | gt=`1973`
  - Q: [image] In which year did the opera house open? Answer with a short direct answer.
  - step0: pred=`unknown` raw=`<|im_end|>` reward=0.000 exact=0
  - final: pred=`unknown` raw=`<|im_end|>` reward=0.000 exact=0
- `visualsimpleqa_da-test-8` | category=sports | gt=`nanjing 2014 youth olympic games`
  - Q: [image] In which sports meeting did this sport make its debut on the Olympic stage? Answer with a short direct answer.
  - step0: pred=`unknown` raw=`<|im_end|>` reward=0.000 exact=0
  - final: pred=`unknown` raw=`<|im_end|>` reward=0.000 exact=0
- `visualsimpleqa_da-test-9` | category=geography | gt=`italy`
  - Q: [image] In which country did this vegetable originate? Answer with a short direct answer.
  - step0: pred=`china` raw=`} China<|im_end|>` reward=0.000 exact=0
  - final: pred=`china` raw=`} China<|im_end|>` reward=0.000 exact=0
- `visualsimpleqa_da-test-10` | category=history | gt=`19th century`
  - Q: [image] In which century was this kind of musical instrument invented? Answer with a short direct answer.
  - step0: pred=`unknown` raw=`<|im_end|>` reward=0.000 exact=0
  - final: pred=`18th century` raw=`} 18th century<|im_end|>` reward=0.000 exact=0
- `visualsimpleqa_da-test-11` | category=history | gt=`19th century`
  - Q: [image] In which century did this type of banana first make its way to America via Toronto? Answer with a short direct answer.
  - step0: pred=`unknown` raw=`<|im_end|>` reward=0.000 exact=0
  - final: pred=`unknown` raw=`<|im_end|>` reward=0.000 exact=0
- `visualsimpleqa_da-test-14` | category=other | gt=`yes`
  - Q: [image] Is this bird a migratory bird? Answer with a short direct answer.
  - step0: pred=`it is not possible to determine from image whether bird is migratory bird or not` raw=` It is not possible to determine from the image whether the bird is a migratory bird or not.<|im_end|>` reward=0.000 exact=0
  - final: pred=`it is not possible to determine from image whether bird is migratory bird or not` raw=` It is not possible to determine from the image whether the bird is a migratory bird or not.<|im_end|>` reward=0.000 exact=0
- `visualsimpleqa_da-test-15` | category=research and education | gt=`google`
  - Q: [image] Which company developed the large language model that shares its name with this tree? Answer with a short direct answer.
  - step0: pred=`unknown` raw=`<|im_end|>` reward=0.000 exact=0
  - final: pred=`unknown` raw=`<|im_end|>` reward=0.000 exact=0
- `visualsimpleqa_da-test-16` | category=film and television entertainment | gt=`1961`
  - Q: [image] In which year was the Disney movie about this pet first released? Answer with a short direct answer.
  - step0: pred=`unknown` raw=`<|im_end|>` reward=0.000 exact=0
  - final: pred=`unknown` raw=`<|im_end|>` reward=0.000 exact=0
- `visualsimpleqa_da-test-17` | category=company and brand | gt=`home furnishings and home goods`
  - Q: [image] What does this brand primarily sell? Answer with a short direct answer.
  - step0: pred=`unknown` raw=`<|im_end|>` reward=0.000 exact=0
  - final: pred=`furniture` raw=`} Furniture<|im_end|>` reward=0.000 exact=0
- `visualsimpleqa_da-test-18` | category=company and brand | gt=`munich germany`
  - Q: [image] In which city is the headquarters of this truck brand? Answer with a short direct answer.
  - step0: pred=`unknown` raw=`?<|im_end|>` reward=0.000 exact=0
  - final: pred=`unknown` raw=`?<|im_end|>` reward=0.000 exact=0

## VisualSimpleQA wrong density pseudo-label examples from train rollouts

- train index `visualsimpleqa_da_20_hq_major_correct-train-13` | category=other | gt=`halloween` | density_peak=`off` | majority=`unknown` | correct_count=0 | valid_ratio=0.25
  - Q: [image] What holiday are the decorations on the left table most associated with? Answer with a short direct answer.
  - counter: `unknown`:24, `reapers`:1, `based on image decorations on left table suggest association with thanksgiving`:1, `turkeys`:1, `off`:1, `semiclassical`:1, `themost associated with decorations on left table is thanksgiving`:1, `b`:1
- train index `visualsimpleqa_da_20_hq_major_correct-train-5` | category=company and brand | gt=`coca cola` | density_peak=`pepsi` | majority=`unknown` | correct_count=0 | valid_ratio=0.28
  - Q: [image] Which cola company uses this animal as its mascot? Answer with a short direct answer.
  - counter: `unknown`:23, `pepsi`:4, `polar bear`:1, `organism`:1, `net energy`:1, `soviet union pepsi`:1, `movetohead`:1
- train index `visualsimpleqa_da_20_hq_major_correct-train-9` | category=art | gt=`united kingdom` | density_peak=`germany` | majority=`united kingdom` | correct_count=6 | valid_ratio=0.91
  - Q: [image] Which country is the author of the famous novel shown on this bus from? Answer with a short direct answer.
  - counter: `united kingdom`:6, `bulgaria`:5, `unknown`:3, `reland`:3, `uk`:2, `poland`:2, `implicit ur`:1, `luxembourg`:1
- train index `visualsimpleqa_da_20_hq_major_correct-train-4` | category=other | gt=`30` | density_peak=`32` | majority=`30` | correct_count=6 | valid_ratio=0.97
  - Q: [image] How many permanent teeth does this animal typically have in adulthood? Answer with a short direct answer.
  - counter: `30`:6, `32`:4, `18`:2, `fifteen`:1, `three`:1, `52`:1, `38 permanent teeth`:1, `starting age of 2 cat typically has 42 permanent teeth`:1
- train index `visualsimpleqa_da_20_hq_major_correct-train-14` | category=other | gt=`japan` | density_peak=`american` | majority=`unknown` | correct_count=4 | valid_ratio=0.69
  - Q: [image] Which country's cuisine is this chef likely cooking? Answer with a short direct answer.
  - counter: `unknown`:10, `japanese`:4, `japan`:4, `armenia`:2, `american`:2, `based on image it s difficult to determine exact country but chef seems to be cooking traditional cuisine that may be associated with east asian southeast asian american mexican or potentially japanese given style of cooking equipment and utensils`:1, `based on image it is not possible to determine chef s nationality or specific country s cuisine from which they are cooking`:1, `country whose culture is brazilian`:1
- train index `visualsimpleqa_da_20_hq_major_correct-train-12` | category=geography | gt=`china` | density_peak=`japan` | majority=`china` | correct_count=8 | valid_ratio=1.00
  - Q: [image] Which country has the most of this animal? Answer with a short direct answer.
  - counter: `china`:8, `poland`:6, `armenia`:3, `bulgaria`:2, `exert`:1, `thailand`:1, `russia`:1, `britain`:1
- train index `visualsimpleqa_da_20_hq_major_correct-train-8` | category=other | gt=`no` | density_peak=`yes it is prohibited` | majority=`no` | correct_count=17 | valid_ratio=1.00
  - Q: [image] In the scene shown in the image, is a vehicle permitted to pass through the arched gate? Answer with a short direct answer.
  - counter: `no`:17, `no vehicle is not permitted to pass through arched gate`:7, `no it is not permitted to pass through gate`:1, `vehicle is not permitted to pass through arched gate`:1, `no because prohibitory road sign on right shows not rentable`:1, `no vehicle is not permitted to pass through`:1, `no vehicles are not permitted to pass through arched gate as indicated by road sign`:1, `sign indicates that vehicles are not allowed to pass through gate`:1
- train index `visualsimpleqa_da_20_hq_major_correct-train-12` | category=geography | gt=`china` | density_peak=`mongolia` | majority=`china` | correct_count=7 | valid_ratio=1.00
  - Q: [image] Which country has the most of this animal? Answer with a short direct answer.
  - counter: `china`:7, `poland`:4, `armenia`:4, `bulgaria`:3, `mongolia`:1, `ukraine`:1, `country with most pandas is china`:1, `estonia japan china`:1
- train index `visualsimpleqa_da_20_hq_major_correct-train-10` | category=research and education | gt=`no` | density_peak=`yes` | majority=`yes` | correct_count=3 | valid_ratio=1.00
  - Q: [image] Do different people always perceive the green vegetables in the picture as having the same smell? Answer with a short direct answer.
  - counter: `yes`:19, `no`:3, `description doesn t provide information on whether different people perceive smell of green vegetables similarly`:1, `it s difficult to say`:1, `concept of smell being perceived is subjective to olfactory receptors of individual and can vary from person to person`:1, `entially`:1, `unanswerable`:1, `answers`:1
- train index `visualsimpleqa_da_20_hq_major_correct-train-4` | category=other | gt=`30` | density_peak=`32` | majority=`30` | correct_count=8 | valid_ratio=0.84
  - Q: [image] How many permanent teeth does this animal typically have in adulthood? Answer with a short direct answer.
  - counter: `30`:8, `unknown`:5, `32`:4, `three`:2, `42`:2, `eight`:2, `animal in image is cat which typically has 30 permanent teeth in adulthood`:1, `between twenty four and thirty in humans`:1

## Working interpretation

1. VisualSimpleQA is dominated by open-world visual + external knowledge questions, not pure extraction. The 20 HQ samples cannot teach broad facts such as brand/company/year/species relations.
2. The density reward often collapses to frequency over short strings because `corr_adv_density_freq` is close to 1.0. It does not add much beyond selecting the most common non-unknown answer.
3. OCRBench benefits because the answer is usually literal text in the image; density/PAPO rewards concise extraction and heavily reduces invalid/blank outputs.
4. A-OKVQA benefits because the answer space is semantically narrower and the official soft/direct-answer metric accepts common short object/action answers.
5. VisualSimpleQA still has many cases where the selected dense answer is plausible but wrong (`Pepsi` vs `Coca-Cola`, wrong year, wrong country), so self-training reinforces confident wrong world knowledge rather than improving visual grounding.