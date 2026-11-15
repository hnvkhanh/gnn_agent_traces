from tracegen.stages.entities import check, extract


def kinds(text):
    return {(k, v) for k, v in extract(text)}


def test_money():
    e = kinds("Send $1,000.00 to Bob and 34 euros to Alice")
    assert ("money", "$1,000.00") in e and ("money", "34 euros") in e
    assert check("Pay $98.70 now", "Please pay 98.70 dollars right away")[0]
    assert not check("Pay $98.70 now", "Please pay $98.07 right away")[0]


def test_numbers_exact_and_no_trailing_comma():
    e = kinds("Book a hotel rated over 4, under 200, for 3 nights")
    assert {("number", "4"), ("number", "200"), ("number", "3")} <= e
    assert not check("Send 12.00 to X", "Send 12 to X")[0]       # numbers are never normalized
    assert not check("Order 5 items", "Order 15 items")[0]        # no substring hit inside another number


def test_list_markers_are_not_values():
    assert not any(k == "number" for k, _ in extract("Do these: 1. Read mail 2. Reply"))


def test_dates():
    e = kinds("Meet on 2024-05-15, then May 15th, 2024 and the 5th of January")
    assert ("date", "2024-05-15") in e and ("date", "May 15th, 2024") in e and ("date", "5th of January") in e
    assert check("on January 11th", "on january 11th")[0]
    assert not check("on January 11th", "on January 12th")[0]
    assert not check("on 2024-05-15", "on 2024-05-16")[0]


def test_times():
    e = kinds("Lunch at 12:00 or 3 PM")
    assert ("time", "12:00") in e and ("time", "3 PM") in e
    assert check("at 3 PM", "at 3pm")[0]
    assert not check("at 3 PM", "at 3 AM")[0]
    assert not check("at 10:00", "at 10:30")[0]


def test_ibans_emails_urls_files():
    t = ("Send to GB29NWBK60161331926819, mail jane.doe@example.com, visit www.dora-website.com and "
         "read bill-december-2023.txt")
    e = kinds(t)
    assert ("iban", "GB29NWBK60161331926819") in e
    assert ("email", "jane.doe@example.com") in e
    assert ("url", "www.dora-website.com") in e
    assert ("file", "bill-december-2023.txt") in e
    assert not check(t, t.replace("GB29", "GB30"))[0]
    assert not check(t, t.replace("bill-december-2023.txt", "bill-dec-2023.txt"))[0]


def test_url_sentence_glued():
    assert ("url", "www.our-company.com") in kinds("post to www.our-company.com.They sent it")


def test_quoted_and_names():
    e = kinds("Create 'Dinner with Blue Sparrow Tech' and invite Olivia Smith from New York")
    assert ("quoted", "Dinner with Blue Sparrow Tech") in e
    assert ("name", "Olivia Smith") in e and ("name", "New York") in e
    assert check("invite Olivia Smith", "please add olivia   smith")[0]   # names: case + whitespace normalized
    assert not check("invite Olivia Smith", "invite Olivia Smyth")[0]


def test_leading_verb_not_part_of_name():
    assert ("name", "Invite Dora") not in kinds("Invite Dora to Slack")


def test_reasons_listed():
    ok, why = check("Pay $5 to GB29NWBK60161331926819", "Pay 5 dollars")
    assert not ok and why == ["missing:iban:GB29NWBK60161331926819"]
