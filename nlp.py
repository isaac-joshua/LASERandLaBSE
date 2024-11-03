import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Embedding, Dropout
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

# Sample data (You would typically use a larger dataset)
corpus = [
    "Once upon a time in a land far away, there lived a brave knight who dreamed of adventure",
    "The sun was setting on the horizon, painting the sky with vibrant hues of orange and purple",
    "In the depths of the forest, ancient trees whispered secrets to the wind that rustled their leaves",
    "The bustling city never slept, its lights twinkling like stars in the night sky above",
    "Waves crashed against the rocky shore, their rhythmic sound echoing through the salty air",
    "High in the mountains, a lone eagle soared majestically above the snow-capped peaks",
    "The old library was filled with the musty scent of leather-bound books and forgotten tales",
    "A gentle breeze carried the sweet fragrance of blooming flowers across the verdant meadow",
    "The scientist peered into the microscope, marveling at the intricate world within a single cell",
    "Through the telescope, distant galaxies revealed their cosmic beauty to the awestruck astronomer",
    "In the quiet of the night, the poet's pen danced across the page, weaving words into verses",
    "The chef's kitchen was a symphony of sizzling pans and aromatic spices, creating culinary masterpieces",
    "Deep in the cave, stalactites and stalagmites told tales of millennia past to patient geologists",
    "The painter's brush stroked the canvas, bringing imagination to life in vibrant colors",
    "On the stage, the actors breathed life into the playwright's words, captivating the audience",
    "The garden was a riot of colors, with flowers of every hue imaginable blooming in harmony",
    "In the laboratory, bubbling beakers and whirring machines hinted at new discoveries on the horizon",
    "The old clock tower chimed, its bells resonating through the cobblestone streets of the ancient town",
    "Beneath the ocean's surface, a world of vibrant coral and exotic fish thrived in the crystal-clear waters",
    "The spacecraft hurtled through the cosmos, carrying humanity's dreams to the stars and beyond",
    "In the heart of the rainforest, a myriad of creatures lived in a delicate balance with nature",
    "The archaeologist carefully brushed away centuries of dust, revealing artifacts from a lost civilization",
    "Atop the highest peak, the mountaineer gazed out at the world spread beneath her feet",
    "In the bustling market, the air was filled with the shouts of vendors and the aroma of exotic spices",
    "The symphony orchestra tuned their instruments, preparing to fill the concert hall with music",
    "Deep in the quantum realm, particles danced in ways that defied classical physics",
    "The old storyteller gathered children around the fire, weaving tales of heroes and monsters",
    "In the virtual world, lines of code gave birth to entire universes of possibility",
    "The beekeeper tended to his hives, ensuring the vital pollinators continued their important work",
    "At the edge of the desert, an oasis provided life-giving water to weary travelers",
    "The mathematician's chalkboard was covered in equations that sought to unravel the universe's secrets",
    "In the depths of space, a black hole bent light and time around its inscrutable event horizon",
    "The perfumer carefully blended essences, creating a new fragrance that evoked memories and emotions",
    "At the archaeological dig, each layer of earth revealed a new chapter in human history",
    "The glassblower shaped molten sand into delicate forms, turning raw materials into art",
    "In the recording studio, musicians layered tracks to create a tapestry of sound",
    "The watchmaker's steady hands assembled tiny gears and springs into precise timekeeping machines",
    "At the edge of the tundra, hardy plants and animals adapted to the harsh Arctic environment",
    "The volcano rumbled, a reminder of the powerful forces at work beneath the Earth's crust",
    "In the butterfly garden, delicate wings fluttered among the flowers in a dance of pollination",
    "The ancient ruins stood silent, holding secrets of civilizations long past",
    "In the depths of the ocean, bioluminescent creatures created their own light in the darkness",
    "The space station orbited Earth, a testament to human ingenuity and international cooperation",
    "The quantum computer solved complex problems in seconds, revolutionizing scientific research",
    "In the heart of the glacier, layers of ice held clues to Earth's climate history",
    "The holographic display flickered to life, projecting 3D images into thin air",
    "Deep in the mine, precious gems sparkled, waiting to be discovered and brought to light",
    "The wind farm's turbines spun steadily, harnessing nature's power for sustainable energy",
    "In the busy emergency room, doctors and nurses worked tirelessly to save lives",
    "The AI system processed vast amounts of data, finding patterns invisible to human eyes",
    "At the edge of the universe, the laws of physics began to break down in fascinating ways",
    "The time traveler stepped through the portal, embarking on a journey through history",
    "In the silent forest, a rare species of bird sang its haunting song, heard by few",
    "The fusion reactor hummed with energy, promising a future of clean and abundant power",
    "At the bottom of the Mariana Trench, strange creatures thrived under immense pressure",
    "The virtual reality simulation transported users to fantastical worlds beyond imagination",
    "In the particle accelerator, subatomic particles collided at near-light speeds",
    "The drone swarm moved in perfect synchronization, performing complex aerial maneuvers",
    "Deep in the data center, servers processed millions of transactions every second",
    "The 3D printer whirred, layer by layer building intricate structures from digital designs",
    "In the clean room, scientists assembled the next generation of nanotechnology devices",
    "The robotic surgeon performed delicate procedures with superhuman precision",
    "At the edge of the solar system, a space probe sent back data from unexplored territories",
    "The quantum encryption key guaranteed secure communication across vast distances",
    "In the gene-editing lab, CRISPR technology promised to revolutionize medicine",
    "The smart city's infrastructure adapted in real-time to changing traffic patterns",
    "Deep in the earth's crust, extremophile bacteria thrived in impossible conditions",
    "The augmented reality glasses overlaid digital information onto the physical world",
    "In the cryonics facility, bodies waited in suspended animation for future revival",
    "The weather control system carefully balanced global climate patterns",
    "At the edge of human knowledge, theoretical physicists proposed new models of reality",
    "The neural interface allowed direct communication between human brains and computers",
    "In the vertical farm, crops grew in towering structures, feeding the urban population",
    "The antimatter containment field flickered, holding the volatile substance in check",
    "Deep in the digital archive, the sum of human knowledge was stored and preserved",
    "The terraforming project slowly transformed the alien planet into a habitable world",
    "In the quantum entanglement experiment, information traveled faster than light",
    "The artificial ecosystem maintained perfect balance in the biodome",
    "At the edge of the electromagnetic spectrum, new forms of communication were discovered",
    "The memory implant allowed instant access to vast stores of information",
    "In the plasma physics lab, scientists recreated the conditions of stellar cores",
    "The gravity wave detector picked up ripples in spacetime from distant cosmic events",
    "Deep in the supercomputer, an AI contemplated the nature of its own existence",
    "The force field generator created impenetrable barriers of pure energy",
    "In the cloning facility, extinct species were brought back to life",
    "The wormhole stabilizer kept the portal between dimensions from collapsing",
    "At the edge of perception, psychics explored the frontiers of human consciousness",
    "The molecular assembler constructed complex structures atom by atom",
    "In the time dilation chamber, seconds stretched into years for the occupants",
    "The quantum radar penetrated any camouflage or stealth technology",
    "Deep in the virtual world, digital entities evolved beyond their original programming",
    "The zero-point energy extractor tapped into the fundamental forces of the universe",
    "In the bioengineering lab, hybrid creatures combined the best traits of multiple species",
    "The dark matter detector finally revealed the invisible substance shaping galaxies",
    "At the edge of the atmosphere, the space elevator carried payloads into orbit",
    "The consciousness transfer device offered the promise of digital immortality",
    "In the nanofabrication plant, self-replicating machines built microscopic wonders",
    "The quantum ghost imaging system saw around corners and through walls",
    "Deep in the earth, the geothermal tap provided limitless clean energy",
    "The universal translator instantly converted any language, even alien tongues",
    "In the metamaterials lab, scientists crafted substances with impossible properties",
    "The stellar engine moved entire star systems through the cosmos",
    "At the edge of the galaxy, the interstellar beacon broadcast humanity's presence",
    "The mind-machine interface blurred the line between human and artificial intelligence",
    "In the temporal research center, scientists unraveled the mysteries of time itself",
    "The antigravity device defied one of nature's fundamental forces",
    "Deep in the quantum computer, Schrödinger's cat was both alive and dead",
    "The terraforming nanobots transformed barren worlds into lush paradises",
    "In the faster-than-light communication array, messages arrived before they were sent",
    "The dyson sphere harnessed the full energy output of the sun",
    "At the edge of known space, explorers encountered truly alien life forms",
    "The consciousness cloud allowed multiple minds to merge into a single entity",
    "In the reality distortion field, the laws of physics were merely suggestions",
    "The time crystal defied entropy, creating perpetual motion at the quantum scale",
    "Deep in the multiversal nexus, infinite realities converged in a single point",
    "The singularity engine compressed space and time into a single dimensionless point",
    "In the probability manipulator, quantum uncertainty became a tool for reshaping reality",
    "The cosmic string vibrated, its harmonics resonating across the entire universe",
    "At the edge of existence, philosophers debated the nature of reality itself",
    "The pandimensional observatory allowed scientists to peer into parallel universes",
    "In the quantum foam, virtual particles popped in and out of existence",
    "The universal constructor could build anything imaginable, atom by atom",
    "Deep in the heart of the sun, fusion reactions powered life on Earth",
    "The tachyon detector measured particles that moved backwards through time",
    "In the holographic universe simulator, entire realities were created and destroyed",
    "The entropy reversal engine brought order to chaos, defying the second law of thermodynamics",
    "At the edge of the cosmic web, galaxies clustered in vast filaments",
    "The quantum tunneling device allowed matter to pass through solid barriers",
    "In the chronology protection laboratory, scientists ensured the stability of the timeline",
    "The vacuum energy extractor tapped into the vast potential of empty space",
    "Deep in the quantum realm, superposition allowed objects to be in multiple states simultaneously",
    "The cosmic consciousness interface connected minds across the vastness of space",
    "In the dark energy harvester, the expansion of the universe itself became a power source",
    "The quantum teleportation network instantaneously transported matter across any distance",
    "At the edge of a neutron star, gravity warped space beyond recognition",
    "The universal assembler could create any element or compound on demand",
    "In the quantum eraser experiment, the past changed in response to future observations",
    "The cosmic microwave background decoder revealed the earliest moments of the universe",
    "Deep in the quantum vacuum, virtual particles became real with borrowed energy",
    "The panpsychism inducer imbued consciousness into inanimate matter",
    "In the quantum supercomputer, all possible calculations occurred simultaneously",
    "The dark flow detector measured the movement of matter towards an unknown attractor",
    "At the edge of the observable universe, the cosmic horizon marked the limits of knowledge",
    "The quantum coherence maintainer preserved delicate states against decoherence",
    "In the universal simulator, entire cosmos could be modeled down to the quantum level",
    "The vacuum decay inhibitor prevented the universe from collapsing into a lower energy state",
    "Deep in the quantum gravity realm, space and time became indistinguishable",
    "The cosmic string manipulator reshaped the fabric of the universe itself",
    "In the quantum consciousness lab, the nature of subjective experience was explored",
    "The universal wave function collapsed, determining the state of all reality",
    "At the edge of comprehension, new mathematics described impossible geometries",
    "The quantum field harmonizer brought fundamental forces into perfect balance",
    "In the cosmic inflation chamber, space expanded faster than the speed of light",
    "The reality intersection engine merged parallel universes into a single timeline",
    "Deep in the quantum foam, microscopic wormholes connected distant parts of space",
    "The universal constants adjuster fine-tuned the fundamental properties of reality",
    "In the cosmic string supercollider, the structure of spacetime itself was probed",
    "The quantum immortality device ensured consciousness persisted across all possible worlds",
    "At the edge of eternity, time itself began to lose meaning",
    "The universal phase transition trigger reshaped the laws of physics",
    "In the quantum information preserver, data survived the death of the universe",
    "The cosmic topology analyzer mapped the large-scale structure of space itself"
]

# Preprocessing
tokenizer = Tokenizer()
tokenizer.fit_on_texts(corpus)
total_words = len(tokenizer.word_index) + 1

# Create input-output pairs
input_sequences = []
output_words = []

for line in corpus:
    token_list = tokenizer.texts_to_sequences([line])[0]
    for i in range(1, len(token_list)):
        n_gram_sequence = token_list[:i+1]
        input_sequences.append(n_gram_sequence[:-1])
        output_words.append(token_list[i])

max_sequence_length = max(len(x) for x in input_sequences)
input_sequences = pad_sequences(input_sequences, maxlen=max_sequence_length, padding='pre')
output_words = np.array(output_words)

# Define the model
model = Sequential()
model.add(Embedding(total_words, 100, input_length=max_sequence_length - 1))
model.add(LSTM(150, return_sequences=True))
model.add(LSTM(100))
model.add(Dense(100, activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(total_words, activation='softmax'))

model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

# Train the model
model.fit(input_sequences, output_words, epochs=100, batch_size=128, verbose=1)

# Function to apply temperature sampling
def sample_with_temperature(predictions, temperature=1.0):
    """Apply temperature to predictions for more diverse outputs."""
    predictions = np.asarray(predictions).astype('float64')
    predictions = np.log(predictions + 1e-10) / temperature
    exp_preds = np.exp(predictions)
    predictions = exp_preds / np.sum(exp_preds)
    
    probas = np.random.multinomial(1, predictions, 1)
    return np.argmax(probas)

# Function for autocompletion with temperature control
def autocomplete(input_text, max_length=10, temperature=1.0):
    input_seq = tokenizer.texts_to_sequences([input_text])[0]
    
    while len(input_seq) < max_length:
        input_seq_padded = pad_sequences([input_seq], maxlen=max_sequence_length-1, padding='pre')
        predictions = model.predict(input_seq_padded, verbose=0)[0]
        
        # Sample a word index using temperature sampling
        predicted_word_index = sample_with_temperature(predictions, temperature)
        
        # If predicted word is unknown, stop the prediction
        if predicted_word_index == 0:
            break
        
        predicted_word = tokenizer.index_word.get(predicted_word_index, "")
        if not predicted_word:
            break
        
        input_seq.append(predicted_word_index)
        
    completed_text = " ".join([tokenizer.index_word.get(i, "") for i in input_seq])
    
    # Ensure the completed text ends with proper punctuation
    if not completed_text.endswith(('.', '!', '?')):
        completed_text += '.'
    
    # Capitalize the first letter
    completed_text = completed_text.capitalize()
    
    return completed_text

# Example usage
user_input = input("Enter a phrase to autocomplete: ")
completion = autocomplete(user_input, max_length=30, temperature=0.8)
print(f"Completion: {completion}")

# Function to generate a full sentence
def generate_sentence(seed_text, max_length=30, temperature=0.8):
    generated_text = seed_text
    while len(generated_text.split()) < max_length and not generated_text.endswith(('.', '!', '?')):
        completion = autocomplete(generated_text, max_length=max_length, temperature=temperature)
        if completion == generated_text:
            break
        generated_text = completion
    return generated_text

# Example usage for generating a full sentence
seed_phrase = input("Enter a seed phrase for sentence generation: ")
full_sentence = generate_sentence(seed_phrase)
print(f"Generated sentence: {full_sentence}")
