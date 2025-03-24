# Problem biznesowy

Aktualnie, z powodu potrzeby manualnego tworzenia playlisty dla grupy, pojawia się spore ryzyko stworzenia playlisty, która przypadnie do gustu wyłącznie jej autorowi. Pojawia się znaczące ryzyko, że aby uniknąć takich problemów, użytkownicy przestaną wykorzystywać Pozytywkę na rzecz aplikacji lepiej dostosowanej do spotkań wieloosobowych.

## Zadanie modelowania

Wykorzystanie funkcji generowania rekomendacji, która dla kontekstu (zsumowanej listy piosenek wszystkich obecnych) zwraca wektor proponowanych utworów. Następnie, za pomocą modelowania sekwencji, jesteśmy w stanie na bieżąco dostosowywać playlistę.

## Specyfikacja

### Kontekst uwzględniany w czasie rzeczywistym:
- Pominięte utwory

### Dane wejściowe:
- Dane o utworach (taneczność, energia, głośność, ilość mowy w utworze, akustyczność, instrumentalność, emocjonalny nastrój utworu, tempo)
- Dane o gatunkach lubianych przez użytkowników i pisanych przez artystów
- Historie słuchania użytkowników (które utwory pomija, częstotliwość odsłuchań, data danego odsłuchania/pominięcia)

### Dane wyjściowe:
- Generowana playlista grupowa (30 utworów z dynamiczną aktualizacją co 5 odtwarzanych utworów)

## model bazowy

1. Modyfikujemy wyniki liczby odsłuchań ostatnich trzydziestu przesłuchanych utworów każdego użytkownika.
2. Łączymy wyniki w jedną listę, uśredniając wagowo wyniki każdego użytkownika.
3. Rekomendujemy wektor utworów generowanych na podstawie uśrednionych wag.
4. Następnie rankinguje rekomendowane utwory względem preferencji każdego użytkownika, eliminując najgorzej rankingowane.
5. Tworzymy wektor o stałej długości n, dostosowując ranking tak, aby unikać utworów kontrowersyjnych.

## Model zaawansowany

1. Po odsłuchaniu bieżącej listy generowane są nowe rekomendacje na podstawie historii pominięć i odtworzeń.
2. Modelowanie sekwencji dostosowuje playlistę do aktualnych preferencji użytkowników.
3. Zalecana ilość użytkowników: [2; 10], lecz rozwiązanie nie ma ograniczeń ilościowych.

## Wejścia i wyjścia

1. **Generowanie rekomendacji**
   - **Wejście:** lista odsłuchanych utworów z informacją o ilości przesłuchań
   - **Wyjście:** lista rekomendowanych utworów
2. **Rankingowanie**
   - **Wejście:** historia polubień i pominięć
   - **Wyjście:** prawdopodobieństwo polubienia utworu
3. **Modelowanie sekwencji**
   - **Wejście:** lista odsłuchanych utworów wraz z reakcjami (skip/play/like)
   - **Wyjście:** lista kolejnych utworów

## Sposób użycia
Aby uruchomić aplikację, skorzystaj z poniższego polecenia
```
    docker compose up --build
```
Po uruchomieniu, Docker zbuduje aplikację, dostępną pod adresem:

http://localhost:8000
