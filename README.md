# Zbiór prefiksów IPv4

Implementacja w C funkcji `add`, `del` i `check` do zarządzania zbiorem prefiksów IPv4.

## Budowanie i testy

```
make check          # kompilacja i testy
make sanitize       # testy z AddressSanitizer i UBSan (wycieki, niezdefiniowane zachowanie)
make unsigned-char  # testy z char bez znaku, jak na ARM
make clean
```

## Struktura danych

Binarne drzewo trie. Każdy poziom drzewa odpowiada jednemu bitowi adresu, od najstarszego.
Prefiks o masce k to węzeł na głębokości k z ustawioną flagą `is_prefix`.

Węzeł nie przechowuje ani bazy, ani maski: baza wynika z drogi od korzenia, a maska z głębokości.
Korzeń jest zmienną statyczną i odpowiada prefiksowi `/0`.

- `add` schodzi po bitach bazy przez `mask` poziomów, tworzy brakujące węzły i ustawia flagę.
- `check` schodzi po bitach adresu, zapamiętuje głębokość ostatniej napotkanej flagi i kończy, gdy nie ma dalszej drogi.
- `del` zdejmuje flagę, a w drodze powrotnej zwalnia węzły bez flagi i bez dzieci.

## Złożoność

| Operacja | Czas  |
|----------|-------|
| `add`    | O(32) |
| `del`    | O(32) |
| `check`  | O(32) |

Czas nie zależy od liczby prefiksów w zbiorze. `check`, wywoływana najczęściej, wykonuje najwyżej 32 kroki w dół drzewa.

## Pamięć

- Węzeł to dwa wskaźniki i flaga: 24 B na platformie 64-bitowej, 12 B na 32-bitowej.
- Dodanie prefiksu tworzy najwyżej `mask` nowych węzłów; prefiksy o wspólnym początku dzielą węzły.
- `del` zwalnia puste węzły, więc po usunięciach drzewo nie trzyma niepotrzebnej pamięci.

## Decyzje projektowe

- Maska spoza zakresu 0..32 to błędny argument (-1). Maska jest od razu zamieniana na `int`, bo `char` może być bez znaku.
- Baza z ustawionymi bitami za maską (np. `10.20.5.0/16`) to błędny argument (-1).
- Usunięcie prefiksu, którego nie ma w zbiorze, zwraca 0, bo argumenty są poprawne.
- Brak pamięci w `add` zwraca -1, a węzły utworzone w tym wywołaniu są zwalniane.
- Kod zakłada 32-bitowy `unsigned int`; inaczej kompilacja kończy się błędem (`#error`).
- Nowe węzły są zerowane ręcznie, a nie przez `calloc`, bo standard C nie gwarantuje, że wskaźnik z samych zer bitów to `NULL`.
- Funkcje pomocnicze: `clear` zwalnia cały zbiór, `node_count` zwraca liczbę węzłów (do testów pamięci).

## Możliwe ulepszenia

- Drzewo Patricia (kompresja ścieżek): około 2 węzłów na prefiks zamiast do 32, kosztem dłuższego kodu `add` i `del`.
- Drzewo wielobitowe (np. 4 bity na poziom): mniej kroków w `check`, ale więcej pamięci i bardziej złożone aktualizacje.
