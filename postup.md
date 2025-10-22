V prvnim kroku se snazime pomoci supervizovaneho uceni dostat model na uroven, kdy je schopen fungovat stejne jako algoritmus pocitajici materialovou situaci.

Trenujeme tedy neuronku o trech vrstvach v radech stovek neuronu (input 133, [256, 512, 128], output size 1), kde vystupem je spojita hodnota urcujici evaluaci pozice

Model trenujeme pomoci nahodne generovanych pozic tak, za minimalizujeme chybu evaluace oproti ohodnoceni podle hodnoty figurek na sachovnici (referencni hodnoty jsou z algoritmu co pocita hodnotu bilych minus cernych figurek)

Ukazal se dobry postup pomoci curriculum learningu - nejdriv pozice s dvema kraly a pouze jednou dalsi figurkou, pak 2 figurky, pak 3 a potom random pozice s 0-32 dalsimi figurkami

Embedding funguje pomoci prevodu sachovnice do vektoru o velikosti 128, kazda figurka ma dva vstupy - konstanty urcujici typ figurky a barvu. Krome toho je v embeddingu jeste hrac na tahu a moznosti rosady