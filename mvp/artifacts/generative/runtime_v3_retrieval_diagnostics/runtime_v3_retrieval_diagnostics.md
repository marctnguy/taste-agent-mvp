# Runtime v3 Retrieval Diagnostics

- Source prompts: H01-H12 from the frozen HITL diagnostic set
- Retrieval mode: full-universe hard constraints, then B3-only or RRF fusion

## H01
Prompt: Recommend me something to watch.
Intent: GENERAL_DISCOVERY
Universe: 259 | post-constraint: 259 | contextual retrieval: False
v2 top-50 overlap: 50 | newly admitted from outside v2 top-50: 0
Top 10 shortlist:
- 1. Seven Samurai (1954) | B3 rank 1 | context rank None | fusion rank None
- 2. Harakiri (1962) | B3 rank 2 | context rank None | fusion rank None
- 3. High and Low (1963) | B3 rank 3 | context rank None | fusion rank None
- 4. Directed by John Ford (1971) | B3 rank 4 | context rank None | fusion rank None
- 5. The Godfather Part II (1974) | B3 rank 5 | context rank None | fusion rank None
- 6. The Godfather (1972) | B3 rank 6 | context rank None | fusion rank None
- 7. Cinema Paradiso (1988) | B3 rank 7 | context rank None | fusion rank None
- 8. Stop Making Sense (1984) | B3 rank 8 | context rank None | fusion rank None
- 9. A Dog's Will (2000) | B3 rank 9 | context rank None | fusion rank None
- 10. The Novices (1970) | B3 rank 10 | context rank None | fusion rank None

## H02
Prompt: I want something melancholic and intimate.
Intent: MOOD_THEME
Universe: 259 | post-constraint: 259 | contextual retrieval: True
v2 top-50 overlap: 27 | newly admitted from outside v2 top-50: 23
Top 10 shortlist:
- 1. Cinema Paradiso (1988) | B3 rank 7 | context rank 14 | fusion rank 1
- 2. Harakiri (1962) | B3 rank 2 | context rank 38 | fusion rank 2
- 3. As I Was Moving Ahead Occasionally I Saw Brief Glimpses of Beauty (2000) | B3 rank 19 | context rank 18 | fusion rank 3
- 4. High and Low (1963) | B3 rank 3 | context rank 65 | fusion rank 4
- 5. Don't Look Down (2008) | B3 rank 76 | context rank 8 | fusion rank 5
- 6. Sex Rider: Wet Highway (1971) | B3 rank 14 | context rank 58 | fusion rank 6
- 7. Once Upon a Time in America (1984) | B3 rank 13 | context rank 74 | fusion rank 7
- 8. A Dog's Will (2000) | B3 rank 9 | context rank 96 | fusion rank 8
- 9. Meander (2021) | B3 rank 144 | context rank 4 | fusion rank 9
- 10. Antichrist (2009) | B3 rank 98 | context rank 12 | fusion rank 10

## H03
Prompt: I want something comforting but not cheesy.
Intent: MOOD_THEME
Universe: 259 | post-constraint: 259 | contextual retrieval: True
v2 top-50 overlap: 27 | newly admitted from outside v2 top-50: 23
Top 10 shortlist:
- 1. Cinema Paradiso (1988) | B3 rank 7 | context rank 14 | fusion rank 1
- 2. The Green Mile (1999) | B3 rank 30 | context rank 21 | fusion rank 2
- 3. Hope (2013) | B3 rank 72 | context rank 3 | fusion rank 3
- 4. Beefcake (1998) | B3 rank 16 | context rank 38 | fusion rank 4
- 5. John Candy: I Like Me (2025) | B3 rank 89 | context rank 1 | fusion rank 5
- 6. As I Was Moving Ahead Occasionally I Saw Brief Glimpses of Beauty (2000) | B3 rank 19 | context rank 44 | fusion rank 6
- 7. A Dog's Will (2000) | B3 rank 9 | context rank 77 | fusion rank 7
- 8. Jacknife (1989) | B3 rank 18 | context rank 54 | fusion rank 8
- 9. Seven Samurai (1954) | B3 rank 1 | context rank 140 | fusion rank 9
- 10. Shape of My Heart (2024) | B3 rank 135 | context rank 2 | fusion rank 10

## H04
Prompt: Give me something slow or alternative in atmosphere/pace but still extremely unsettling and anxiety-inducing, like Jeanne Dielman.
Intent: GENERAL_DISCOVERY
Universe: 259 | post-constraint: 259 | contextual retrieval: True
v2 top-50 overlap: 26 | newly admitted from outside v2 top-50: 24
Top 10 shortlist:
- 1. The Novices (1970) | B3 rank 10 | context rank 5 | fusion rank 1
- 2. High and Low (1963) | B3 rank 3 | context rank 15 | fusion rank 2
- 3. Psycho (1960) | B3 rank 29 | context rank 10 | fusion rank 3
- 4. As I Was Moving Ahead Occasionally I Saw Brief Glimpses of Beauty (2000) | B3 rank 19 | context rank 20 | fusion rank 4
- 5. Sex Rider: Wet Highway (1971) | B3 rank 14 | context rank 31 | fusion rank 5
- 6. Behold a Pale Horse (1964) | B3 rank 12 | context rank 40 | fusion rank 6
- 7. Cinema Paradiso (1988) | B3 rank 7 | context rank 54 | fusion rank 7
- 8. Castle Freak (1996) | B3 rank 51 | context rank 9 | fusion rank 8
- 9. Harakiri (1962) | B3 rank 2 | context rank 104 | fusion rank 9
- 10. Another Day (2026) | B3 rank 122 | context rank 2 | fusion rank 10

## H05
Prompt: I want something quiet, contemplative and character-driven.
Intent: MOOD_THEME
Universe: 259 | post-constraint: 259 | contextual retrieval: True
v2 top-50 overlap: 26 | newly admitted from outside v2 top-50: 24
Top 10 shortlist:
- 1. Harakiri (1962) | B3 rank 2 | context rank 4 | fusion rank 1
- 2. High and Low (1963) | B3 rank 3 | context rank 9 | fusion rank 2
- 3. A Silent Voice: The Movie (2016) | B3 rank 34 | context rank 12 | fusion rank 3
- 4. Cinema Paradiso (1988) | B3 rank 7 | context rank 58 | fusion rank 4
- 5. As I Was Moving Ahead Occasionally I Saw Brief Glimpses of Beauty (2000) | B3 rank 19 | context rank 38 | fusion rank 5
- 6. Seven Samurai (1954) | B3 rank 1 | context rank 129 | fusion rank 6
- 7. Castle Freak (1996) | B3 rank 51 | context rank 23 | fusion rank 7
- 8. Sex Rider: Wet Highway (1971) | B3 rank 14 | context rank 73 | fusion rank 8
- 9. Once Upon a Time in America (1984) | B3 rank 13 | context rank 77 | fusion rank 9
- 10. Remarkably Bright Creatures (2026) | B3 rank 187 | context rank 1 | fusion rank 10

## H06
Prompt: Show me something different from what I normally watch.
Intent: NOVELTY
Universe: 259 | post-constraint: 259 | contextual retrieval: True
v2 top-50 overlap: 24 | newly admitted from outside v2 top-50: 26
Top 10 shortlist:
- 1. As I Was Moving Ahead Occasionally I Saw Brief Glimpses of Beauty (2000) | B3 rank 19 | context rank 1 | fusion rank 1
- 2. Directed by John Ford (1971) | B3 rank 4 | context rank 26 | fusion rank 2
- 3. Dexter's Laboratory: Ego Trip (1999) | B3 rank 23 | context rank 15 | fusion rank 3
- 4. Regular Show: The Movie (2015) | B3 rank 70 | context rank 2 | fusion rank 4
- 5. High and Low (1963) | B3 rank 3 | context rank 70 | fusion rank 5
- 6. 28 Up (1984) | B3 rank 17 | context rank 36 | fusion rank 6
- 7. Seven Samurai (1954) | B3 rank 1 | context rank 84 | fusion rank 7
- 8. 35 Up (1991) | B3 rank 35 | context rank 24 | fusion rank 8
- 9. Harakiri (1962) | B3 rank 2 | context rank 137 | fusion rank 9
- 10. Beefcake (1998) | B3 rank 16 | context rank 66 | fusion rank 10

## H07
Prompt: I want to be mind-blown, recommend me something totally out there that surprises me.
Intent: GENERAL_DISCOVERY
Universe: 259 | post-constraint: 259 | contextual retrieval: True
v2 top-50 overlap: 21 | newly admitted from outside v2 top-50: 29
Top 10 shortlist:
- 1. Piece by Piece (2024) | B3 rank 80 | context rank 11 | fusion rank 1
- 2. As I Was Moving Ahead Occasionally I Saw Brief Glimpses of Beauty (2000) | B3 rank 19 | context rank 57 | fusion rank 2
- 3. Ghost Elephants (2026) | B3 rank 86 | context rank 12 | fusion rank 3
- 4. Don't Look Down (2008) | B3 rank 76 | context rank 15 | fusion rank 4
- 5. Billie Eilish - Hit Me Hard and Soft: The Tour (Live in 3D) (2026) | B3 rank 186 | context rank 1 | fusion rank 5
- 6. Ocean with David Attenborough (2025) | B3 rank 148 | context rank 5 | fusion rank 6
- 7. Remarkably Bright Creatures (2026) | B3 rank 187 | context rank 2 | fusion rank 7
- 8. Stop Making Sense (1984) | B3 rank 8 | context rank 126 | fusion rank 8
- 9. High and Low (1963) | B3 rank 3 | context rank 187 | fusion rank 9
- 10. Harakiri (1962) | B3 rank 2 | context rank 217 | fusion rank 10

## H08
Prompt: I want a French film tonight.
Intent: CONSTRAINT
Universe: 259 | post-constraint: 6 | contextual retrieval: False
v2 top-50 overlap: 1 | newly admitted from outside v2 top-50: 5
Top 10 shortlist:
- 1. The Novices (1970) | B3 rank 1 | context rank None | fusion rank None
- 2. Call My Agent! The Movie (2026) | B3 rank 2 | context rank None | fusion rank None
- 3. Another Day (2026) | B3 rank 3 | context rank None | fusion rank None
- 4. Meander (2021) | B3 rank 4 | context rank None | fusion rank None
- 5. A Woman's Life (2026) | B3 rank 5 | context rank None | fusion rank None
- 6. You+Me - Against the World (2026) | B3 rank 6 | context rank None | fusion rank None

## H09
Prompt: I want a European movie that feels like freedom and youthful.
Intent: GENERAL_DISCOVERY
Universe: 259 | post-constraint: 259 | contextual retrieval: True
v2 top-50 overlap: 25 | newly admitted from outside v2 top-50: 25
Top 10 shortlist:
- 1. Cinema Paradiso (1988) | B3 rank 7 | context rank 3 | fusion rank 1
- 2. Once Upon a Time in America (1984) | B3 rank 13 | context rank 11 | fusion rank 2
- 3. High and Low (1963) | B3 rank 3 | context rank 58 | fusion rank 3
- 4. As I Was Moving Ahead Occasionally I Saw Brief Glimpses of Beauty (2000) | B3 rank 19 | context rank 30 | fusion rank 4
- 5. The Novices (1970) | B3 rank 10 | context rank 51 | fusion rank 5
- 6. The Good, the Bad and the Ugly (1966) | B3 rank 24 | context rank 37 | fusion rank 6
- 7. Seven Samurai (1954) | B3 rank 1 | context rank 113 | fusion rank 7
- 8. Don't Look Down (2008) | B3 rank 76 | context rank 8 | fusion rank 8
- 9. Behold a Pale Horse (1964) | B3 rank 12 | context rank 66 | fusion rank 9
- 10. Another Day (2026) | B3 rank 122 | context rank 2 | fusion rank 10

## H10
Prompt: I want a good old classic, but no drama.
Intent: CONSTRAINT
Universe: 259 | post-constraint: 164 | contextual retrieval: True
v2 top-50 overlap: 22 | newly admitted from outside v2 top-50: 28
Top 10 shortlist:
- 1. Sex Rider: Wet Highway (1971) | B3 rank 3 | context rank 4 | fusion rank 1
- 2. The Good, the Bad and the Ugly (1966) | B3 rank 8 | context rank 6 | fusion rank 2
- 3. Cavegirl (1985) | B3 rank 17 | context rank 1 | fusion rank 3
- 4. Psycho (1960) | B3 rank 11 | context rank 10 | fusion rank 4
- 5. Directed by John Ford (1971) | B3 rank 1 | context rank 49 | fusion rank 5
- 6. Practical Magic (1998) | B3 rank 47 | context rank 3 | fusion rank 6
- 7. Zero Woman 2 (1995) | B3 rank 16 | context rank 25 | fusion rank 7
- 8. The Empire Strikes Back (1980) | B3 rank 15 | context rank 37 | fusion rank 8
- 9. The Transformers: The Movie (1986) | B3 rank 7 | context rank 61 | fusion rank 9
- 10. Pinocchio: Unstrung (2026) | B3 rank 49 | context rank 12 | fusion rank 10

## H11
Prompt: I just watched La Bola Negra and loved the Penelope Cruz character in it, I want something about performers, fame and/or show business preferably with a period setting.
Intent: GENERAL_DISCOVERY
Universe: 259 | post-constraint: 259 | contextual retrieval: True
v2 top-50 overlap: 27 | newly admitted from outside v2 top-50: 23
Top 10 shortlist:
- 1. Behold a Pale Horse (1964) | B3 rank 12 | context rank 6 | fusion rank 1
- 2. The Novices (1970) | B3 rank 10 | context rank 18 | fusion rank 2
- 3. The Good, the Bad and the Ugly (1966) | B3 rank 24 | context rank 11 | fusion rank 3
- 4. Cinema Paradiso (1988) | B3 rank 7 | context rank 31 | fusion rank 4
- 5. Kiss of the Spider Woman (1985) | B3 rank 54 | context rank 1 | fusion rank 5
- 6. Once Upon a Time in America (1984) | B3 rank 13 | context rank 32 | fusion rank 6
- 7. Dance of the Forty One (2020) | B3 rank 63 | context rank 5 | fusion rank 7
- 8. Don't Look Down (2008) | B3 rank 76 | context rank 4 | fusion rank 8
- 9. High and Low (1963) | B3 rank 3 | context rank 85 | fusion rank 9
- 10. The Godfather Part II (1974) | B3 rank 5 | context rank 77 | fusion rank 10

## H12
Prompt: Based on what you know about my taste, recommend something I might not discover on my own.
Intent: GENERAL_DISCOVERY
Universe: 259 | post-constraint: 259 | contextual retrieval: True
v2 top-50 overlap: 24 | newly admitted from outside v2 top-50: 26
Top 10 shortlist:
- 1. As I Was Moving Ahead Occasionally I Saw Brief Glimpses of Beauty (2000) | B3 rank 19 | context rank 1 | fusion rank 1
- 2. John Candy: I Like Me (2025) | B3 rank 89 | context rank 2 | fusion rank 2
- 3. High and Low (1963) | B3 rank 3 | context rank 87 | fusion rank 3
- 4. Piece by Piece (2024) | B3 rank 80 | context rank 7 | fusion rank 4
- 5. 28 Up (1984) | B3 rank 17 | context rank 52 | fusion rank 5
- 6. Stop Making Sense (1984) | B3 rank 8 | context rank 95 | fusion rank 6
- 7. Directed by John Ford (1971) | B3 rank 4 | context rank 131 | fusion rank 7
- 8. The Sheep Detectives (2026) | B3 rank 149 | context rank 4 | fusion rank 8
- 9. Seven Samurai (1954) | B3 rank 1 | context rank 195 | fusion rank 9
- 10. McQueen (2018) | B3 rank 101 | context rank 11 | fusion rank 10
