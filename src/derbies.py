"""
Derbies and big rivalries in our leagues.

A derby is a match between clubs from the same city or region, or a famous rivalry
(e.g. Arsenal - Tottenham, Sevilla - Betis, Wisla - Cracovia). Names as in football-data.co.uk.

Walk-forward test (2021-22 to 2025-26, 590 derbies): derbies had about 0.14 fewer goals than the
model expected, and BTTS slightly less often. Adding "derby" to the goals model made BTTS a
little better (log loss 0.6874 -> 0.6872). For the match result (1X2) it changed nothing,
so it is used only in the goals model. The website also marks derbies with a badge.
"""

DERBIES = {
    "E0": [("Arsenal", "Tottenham"), ("Arsenal", "Chelsea"), ("Chelsea", "Tottenham"), ("West Ham", "Tottenham"),
           ("West Ham", "Chelsea"), ("West Ham", "Arsenal"), ("Fulham", "Chelsea"), ("Fulham", "Brentford"),
           ("Crystal Palace", "Brighton"), ("Liverpool", "Everton"), ("Man City", "Man United"), ("Man United", "Liverpool"),
           ("Newcastle", "Sunderland"), ("Wolves", "West Brom"), ("Nott'm Forest", "Leicester"), ("Nottm Forest", "Leicester"),
           ("Norwich", "Ipswich"), ("Southampton", "Bournemouth")],
    "SP1": [("Real Madrid", "Ath Madrid"), ("Real Madrid", "Barcelona"), ("Barcelona", "Espanol"), ("Sevilla", "Betis"),
            ("Ath Bilbao", "Sociedad"), ("Ath Bilbao", "Alaves"), ("Ath Bilbao", "Osasuna"), ("Valencia", "Levante"),
            ("Valencia", "Villarreal"), ("Real Madrid", "Getafe"), ("Ath Madrid", "Getafe"), ("Real Madrid", "Vallecano"),
            ("Ath Madrid", "Vallecano"), ("Celta", "La Coruna"), ("Sp Gijon", "Oviedo"), ("Granada", "Malaga"),
            ("Cadiz", "Sevilla"), ("Cadiz", "Betis")],
    "I1": [("Inter", "Milan"), ("Roma", "Lazio"), ("Juventus", "Torino"), ("Genoa", "Sampdoria"), ("Inter", "Juventus"),
           ("Napoli", "Roma"), ("Verona", "Chievo"), ("Atalanta", "Brescia"), ("Fiorentina", "Juventus"),
           ("Empoli", "Fiorentina"), ("Pisa", "Fiorentina"), ("Sassuolo", "Bologna"), ("Parma", "Bologna")],
    "D1": [("Dortmund", "Schalke 04"), ("Bayern Munich", "Dortmund"), ("Hamburg", "Werder Bremen"), ("Hamburg", "St Pauli"),
           ("FC Koln", "M'gladbach"), ("FC Koln", "MGladbach"), ("FC Koln", "Leverkusen"), ("FC Koln", "Fortuna Dusseldorf"),
           ("M'gladbach", "Fortuna Dusseldorf"), ("MGladbach", "Fortuna Dusseldorf"), ("Hertha", "Union Berlin"),
           ("Nurnberg", "Greuther Furth"), ("Ein Frankfurt", "Mainz"), ("Bochum", "Dortmund"), ("Bochum", "Schalke 04"),
           ("Stuttgart", "Freiburg"), ("Stuttgart", "Hoffenheim"), ("Leverkusen", "M'gladbach"), ("Leverkusen", "MGladbach"),
           ("Werder Bremen", "Hannover"), ("Hannover", "Hamburg")],
    "F1": [("Paris SG", "Marseille"), ("Lyon", "St Etienne"), ("Nice", "Monaco"), ("Lens", "Lille"), ("Rennes", "Nantes"),
           ("Rennes", "Brest"), ("Rennes", "Lorient"), ("Rennes", "Guingamp"), ("Brest", "Lorient"), ("Brest", "Guingamp"),
           ("Lorient", "Guingamp"), ("Paris SG", "Paris FC"), ("Bordeaux", "Toulouse"), ("Metz", "Nancy"),
           ("Nimes", "Montpellier"), ("Ajaccio", "Bastia"), ("Lyon", "Marseille"), ("Marseille", "Nice"),
           ("Le Havre", "Caen"), ("Reims", "Troyes"), ("Metz", "Strasbourg")],
    "POL": [("Legia", "Lech Poznan"), ("Wisla", "Cracovia"), ("Lechia Gdansk", "Arka Gdynia"), ("Gornik Zabrze", "Ruch Chorzow"),
            ("Gornik Z.", "Ruch"), ("Gornik Zabrze", "Ruch"), ("Gornik Z.", "Ruch Chorzow"), ("Gornik Zabrze", "Piast Gliwice"),
            ("Gornik Z.", "Piast Gliwice"), ("Piast Gliwice", "GKS Katowice"), ("Gornik Zabrze", "GKS Katowice"),
            ("Gornik Z.", "GKS Katowice"), ("Ruch", "GKS Katowice"), ("Ruch Chorzow", "GKS Katowice"), ("Ruch", "Piast Gliwice"),
            ("Ruch Chorzow", "Piast Gliwice"), ("Widzew Lodz", "LKS Lodz"), ("Lech Poznan", "Warta Poznan"),
            ("Zaglebie", "Slask Wroclaw"), ("Legia", "Widzew Lodz"), ("Lech Poznan", "Widzew Lodz"),
            ("Zaglebie Sosnowiec", "GKS Katowice"), ("Zaglebie Sosnowiec", "Ruch"), ("Pogon Szczecin", "Lech Poznan"),
            ("Termalica B-B.", "Sandecja Nowy S."), ("Korona Kielce", "Radomiak Radom"), ("Stal Mielec", "Termalica B-B.")],
}
PAIRS = {frozenset(pair) for pairs in DERBIES.values() for pair in pairs}


def is_derby(home_team, away_team):
    """True if the two teams are a derby (the order does not matter)."""
    return frozenset((home_team, away_team)) in PAIRS
