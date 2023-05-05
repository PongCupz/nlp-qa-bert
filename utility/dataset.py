from datasets import Dataset

custom_dataset = []
Context = {}

def find_answer_start(context,answer):
    start_aswer = 0
    ans = answer.split()
    cw = []
    found_ans = ""
    for item in ans:
        cw.append(item)
        if len(cw) > 1 :
            x = " ".join(cw)
        else: 
            x = item

        n = context.find(x)
        if n != -1 :
            start_aswer = n
            found_ans = x
    return start_aswer,found_ans

def answer_start(context,answer):
    answer = answer.lower()
    answer = answer.replace(".", "")
    max_len = 0
    result_ans = 0
    while True :
        start_aswer,found_ans = find_answer_start(context.lower(),answer)

        if len(found_ans.split()) > max_len :
            result_ans = start_aswer

        if answer == found_ans :
            break
        if answer.find(' ') != -1 :
            answer = answer.split(' ', 1)[1]
        else :
            break

    return result_ans

### 1
Context["TheGrasshopper"] = '''One bright day in late autumn a family of Ants were bustling about in the warm sunshine, drying out the grain they had stored up during the summer, when a starving Grasshopper, his fiddle under his arm, came up and humbly begged for a bite to eat.
"What!" cried the Ants in surprise, "haven't you stored anything away for the winter? What in the world were you doing all last summer?"
"I didn't have time to store up any food," whined the Grasshopper; "I was so busy making music that before I knew it the summer was gone."
The Ants shrugged their shoulders in disgust.
"Making music, were you?" they cried. "Very well; now dance!" And they turned their backs on the Grasshopper and went on with their work.
There's a time for work and a time for play.'''

### 2
Context["TheShepherdBoy"] = '''A Shepherd Boy tended his master's Sheep near a dark forest not far from the village. Soon he found life in the pasture very dull. All he could do to amuse himself was to talk to his dog or play on his shepherd's pipe.
One day as he sat watching the Sheep and the quiet forest, and thinking what he would do should he see a Wolf, he thought of a plan to amuse himself.
His Master had told him to call for help should a Wolf attack the flock, and the Villagers would drive it away. So now, though he had not seen anything that even looked like a Wolf, he ran toward the village shouting at the top of his voice, "Wolf! Wolf!"
As he expected, the Villagers who heard the cry dropped their work and ran in great excitement to the pasture. But when they got there they found the Boy doubled up with laughter at the trick he had played on them.
A few days later the Shepherd Boy again shouted, "Wolf! Wolf!" Again the Villagers ran to help him, only to be laughed at again.
Then one evening as the sun was setting behind the forest and the shadows were creeping out over the pasture, a Wolf really did spring from the underbrush and fall upon the Sheep.
In terror the Boy ran toward the village shouting "Wolf! Wolf!" But though the Villagers heard the cry, they did not run to help him as they had before. "He cannot fool us again," they said.
The Wolf killed a great many of the Boy's sheep and then slipped away into the forest.
Liars are not believed even when they speak the truth.
Also referred to as: Never cry "Wolf"!'''

### 3
Context["TheCrow"] = '''In a spell of dry weather, when the Birds could find very little to drink, a thirsty Crow found a pitcher with a little water in it. But the pitcher was high and had a narrow neck, and no matter how he tried, the Crow could not reach the water. The poor thing felt as if he must die of thirst.
Then an idea came to him. Picking up some small pebbles, he dropped them into the pitcher one by one. With each pebble the water rose a little higher until at last it was near enough so he could drink.
In a pinch, a good use of our wits may help us out.'''

### 4
Context["TheDog"] = '''A Dog, to whom the butcher had thrown a bone, was hurrying home with his prize as fast as he could go. As he crossed a narrow footbridge, he happened to look down and saw himself reflected in the quiet water as if in a mirror. But the greedy Dog thought he saw a real Dog carrying a bone much bigger than his own.
If he had stopped to think he would have known better. But instead of thinking, he dropped his bone and sprang at the Dog in the river, only to find himself swimming for dear life to reach the shore. At last he managed to scramble out, and as he stood sadly thinking about the good bone he had lost, he realized what a stupid Dog he had been.
It is very foolish to be greedy.'''

### 5
Context["TheBundle"] = '''A certain Father had a family of Sons, who were forever quarreling among themselves. No words he could say did the least good, so he cast about in his mind for some very striking example that should make them see that discord would lead them to misfortune.
One day when the quarreling had been much more violent than usual and each of the Sons was moping in a surly manner, he asked one of them to bring him a bundle of sticks. Then handing the bundle to each of his Sons in turn he told them to try to break it. But although each one tried his best, none was able to do so.
The Father then untied the bundle and gave the sticks to his Sons to break one by one. This they did very easily.
"My Sons," said the Father, "do you not see how certain it is that if you agree with each other and help each other, it will be impossible for your enemies to injure you? But if you are divided among yourselves, you will be no stronger than a single stick in that bundle."
In unity is strength.'''

### 6
Context["TheFrogs"] = '''The Frogs were tired of governing themselves. They had so much freedom that it had spoiled them, and they did nothing but sit around croaking in a bored manner and wishing for a government that could entertain them with the pomp and display of royalty, and rule them in a way to make them know they were being ruled. No milk and water government for them, they declared. So they sent a petition to Jupiter asking for a king.
Jupiter saw what simple and foolish creatures they were, but to keep them quiet and make them think they had a king he threw down a huge log, which fell into the water with a great splash. The Frogs hid themselves among the reeds and grasses, thinking the new king to be some fearful giant. But they soon discovered how tame and peaceable King Log was. In a short time the younger Frogs were using him for a diving platform, while the older Frogs made him a meeting place, where they complained loudly to Jupiter about the government.
To teach the Frogs a lesson the ruler of the gods now sent a Crane to be king of Frogland. The Crane proved to be a very different sort of king from old King Log. He gobbled up the poor Frogs right and left and they soon saw what fools they had been. In mournful croaks they begged Jupiter to take away the cruel tyrant before they should all be destroyed.
"How now!" cried Jupiter "Are you not yet content? You have what you asked for and so you have only yourselves to blame for your misfortunes."
Be sure you can better your condition before you seek to change.'''

### 7
Context["TheLion"] = '''A Lion lay asleep in the forest, his great head resting on his paws. A timid little Mouse came upon him unexpectedly, and in her fright and haste to get away, ran across the Lion's nose. Roused from his nap, the Lion laid his huge paw angrily on the tiny creature to kill her.
"Spare me!" begged the poor Mouse. "Please let me go and some day I will surely repay you."
The Lion was much amused to think that a Mouse could ever help him. But he was generous and finally let the Mouse go.
Some days later, while stalking his prey in the forest, the Lion was caught in the toils of a hunter's net. Unable to free himself, he filled the forest with his angry roaring. The Mouse knew the voice and quickly found the Lion struggling in the net. Running to one of the great ropes that bound him, she gnawed it until it parted, and soon the Lion was free.
"You laughed when I said I would repay you," said the Mouse. "Now you see that even a Mouse can help a Lion."
A kindness is never wasted.'''

### 8
Context["TheFrogs2"] = '''An Ox came down to a reedy pool to drink. As he splashed heavily into the water, he crushed a young Frog into the mud. The old Frog soon missed the little one and asked his brothers and sisters what had become of him.
"A great big monster," said one of them, "stepped on little brother with one of his huge feet!"
"Big, was he!" said the old Frog, puffing herself up. "Was he as big as this?"
"Oh, much bigger!" they cried.
The Frog puffed up still more.
"He could not have been bigger than this," she said. But the little Frogs all declared that the monster was much, much bigger and the old Frog kept puffing herself out more and more until, all at once, she burst.
Do not attempt the impossible.'''

### 9
Context["TheTortoise"] = '''A Hare was making fun of the Tortoise one day for being so slow.
"Do you ever get anywhere?" he asked with a mocking laugh.
"Yes," replied the Tortoise, "and I get there sooner than you think. I'll run you a race and prove it."
The Hare was much amused at the idea of running a race with the Tortoise, but for the fun of the thing he agreed. So the Fox, who had consented to act as judge, marked the distance and started the runners off.
The Hare was soon far out of sight, and to make the Tortoise feel very deeply how ridiculous it was for him to try a race with a Hare, he lay down beside the course to take a nap until the Tortoise should catch up.
The Tortoise meanwhile kept going slowly but steadily, and, after a time, passed the place where the Hare was sleeping. But the Hare slept on very peacefully; and when at last he did wake up, the Tortoise was near the goal. The Hare now ran his swiftest, but he could not overtake the Tortoise in time.
Slow and steady wins the race.'''


### 10
Context["TheYoungCrab"] = '''""Why in the world do you walk sideways like that?" said a Mother Crab to her son. "You should always walk straight forward with your toes turned out."
"Show me how to walk, mother dear," answered the little Crab obediently, "I want to learn."
So the old Crab tried and tried to walk straight forward. But she could walk sideways only, like her son. And when she wanted to turn her toes out she tripped and fell on her nose.
Do not tell others how to act unless you can set a good example.'''

contexts = [
    Context["TheGrasshopper"],Context["TheGrasshopper"],Context["TheGrasshopper"],
    Context["TheShepherdBoy"],Context["TheShepherdBoy"],Context["TheShepherdBoy"],
    Context["TheCrow"],Context["TheCrow"],Context["TheCrow"],
    Context["TheDog"],Context["TheDog"],Context["TheDog"],
    Context["TheBundle"],Context["TheBundle"],Context["TheBundle"],
    Context["TheFrogs"],Context["TheFrogs"],Context["TheFrogs"],
    Context["TheLion"],Context["TheLion"],Context["TheLion"],
    Context["TheFrogs2"],Context["TheFrogs2"],Context["TheFrogs2"],
    Context["TheTortoise"],Context["TheTortoise"],Context["TheTortoise"],
    Context["TheYoungCrab"],Context["TheYoungCrab"],Context["TheYoungCrab"],
]
titles = [
    "TheGrasshopper","TheGrasshopper","TheGrasshopper",
    "TheShepherdBoy","TheShepherdBoy","TheShepherdBoy",
    "TheCrow","TheCrow","TheCrow",
    "TheDog","TheDog","TheDog",
    "TheBundle","TheBundle","TheBundle",
    "TheFrogs","TheFrogs","TheFrogs",
    "TheLion","TheLion","TheLion",
    "TheFrogs2","TheFrogs2","TheFrogs2",
    "TheTortoise","TheTortoise","TheTortoise",
    "TheYoungCrab","TheYoungCrab","TheYoungCrab"
]
questions = [
    "Which season do ants store food?","What does the grasshopper do in all summer?","What did the Ants tell the Grasshopper to do?",
    "Who tended the master's sheep?","What lies did the boy tell?","What happened when the wolf actually attacked the sheep?",
    "What did the crow want to do with the pitcher?","Did the crow succeed in getting a drink from the pitcher?","How does the crow solve the problem of the water being too low in the pitcher?",
    "What was in the dog's mouth?","What did the dog see?","Why did the dog open his mouth?",
    "Who are the main characters in this story?","What was the problem in the family of Sons?","What did the Father tell his Sons after they broke the sticks one by one?",
    "list all their king", "Why were the Frogs bored?","what is moral of this story",
    "Where did the Lion lay asleep?","What did the Mouse beg of the Lion?","What happened to the Lion some days later in the forest?",
    "What happened to the young Frog?","What did the old Frog do when she heard about the monster?","What happened to the old Frog?",
    "Who participated in the race?","Who was the race winner?","Why did the hare lose the competition?",
    "What animal is the main character?","What does the mother crab tell her son about walking?","What does the young crab ask his mother?"
]

answers = [
    "summer","He was too busy making music","They told him to dance.",
    "A Shepherd Boy.",'''He thought of a plan to trick the Villagers by shouting "Wolf! Wolf!" even though there's no wolf.''',"No one believe him",
    "The crow wanted to drink from the pitcher.","Yes","The crow drops small pebbles into the pitcher until the water level to rise until it is high enough for the crow to drink.",
    "a bone","his shadow","He wanted a bone bigger than his own.",
    "the Father and his Sons.","The Sons were forever quarreling among themselves.",'''The Father said, "My Sons, do you not see how certain it is that if you agree with each other and help each other, it will be impossible for your enemies to injure you? But if you are divided among yourselves, you will be no stronger than a single stick in that bundle."''',
    "King Log and the Crane.","They had too much freedom.","Be sure you can better your condition before you seek to change.",
    "the forest","The Mouse begged the Lion to spare her.","The Lion was caught in the toils of a hunter's net.",
    "The young Frog was crushed by the Ox.","The old Frog puffed herself up.","The old Frog burst.",
    "The Tortoise and the Hare","The tortoise","he took a nap during the race",
    "crabs","The mother crab tells her son that he should always walk straight forward","he asks his mother to show him how to walk."
]
for i in range(len(contexts)):
    custom_dataset.append({
        "question" : questions[i],
        "context" : contexts[i],
        "answer" :answers[i]
    })


answers_text = []
for item in answers:
    answers_text.append([item])

answers_start = []
for i in range(len(contexts)):
    answers_start.append([answer_start(contexts[i],answers[i])])

_answers = []
for i in range(len(answers)):
    _answers.append({
        "text":answers_text[i],
        "answer_start":answers_start[i]
    })

ids = []
for i in range(30):
    ids.append(f'customedata{i}')
custom_datasets = {}
custom_datasets['train'] = Dataset.from_dict({'id':ids[0:25],'title':titles[0:25],'context':contexts[0:25],'question':questions[0:25], 'answers':_answers[0:25]})
custom_datasets['validation'] = Dataset.from_dict({'id':ids[25:30],'title':titles[25:30],'context':contexts[25:30],'question':questions[25:30], 'answers':_answers[25:30]})
custom_datasets['all'] = Dataset.from_dict({'id':ids,'title':titles,'context':contexts,'question':questions, 'answers':_answers})

factoid = [1,2,3,4,9,10,11,15,16,18,19,20,21,22,23,24,25,26,29,30]
nonfactoid = [5,6,7,8,12,13,14,17,27,28]

_ids = []
_titles = []
_contexts = []
_questions = []
_answer = []
for i in factoid :
    _ids.append(ids[i-1])
    _titles.append(titles[i-1])
    _contexts.append(contexts[i-1])
    _questions.append(questions[i-1])
    _answer.append(_answers[i-1])
custom_datasets['factoid'] = Dataset.from_dict({'id':_ids,'title':_titles,'context':_contexts,'question':_questions, 'answers':_answer})

_ids = []
_titles = []
_contexts = []
_questions = []
_answer = []
for i in nonfactoid :
    _ids.append(ids[i-1])
    _titles.append(titles[i-1])
    _contexts.append(contexts[i-1])
    _questions.append(questions[i-1])
    _answer.append(_answers[i-1])
custom_datasets['nonfactoid'] = Dataset.from_dict({'id':_ids,'title':_titles,'context':_contexts,'question':_questions, 'answers':_answer})
