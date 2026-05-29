class Book:
    def __init__(self, isbn):
        self.isbn = isbn
        self.bought_copies = 0
        self.bought_price = 0.0
        self.sold_copies = 0
        self.sold_price = 0.0

def month_to_string(month):
    h = {1: 'January', 2: 'February', 3: 'March', 4: 'April', 5: 'May', 6: 'June', 7: 'July', 8: 'August',
         9: 'September', 10: 'October', 11: 'November', 12: 'December'}
    return h[month]

if __name__ == "__main__":

    file = open("Data\ex4_data.txt")

    month_sold_books = {}
    books = {}

    for line in file:
        isbn, transaction, date, copies, price = line.split()
        copies = int(copies)
        price = float(price)

        if isbn not in books:
            books[isbn] = Book(isbn)

        if transaction == "B":
            books[isbn].bought_copies += copies
            books[isbn].bought_price += price * copies
        elif transaction == "S":
            books[isbn].sold_copies += copies
            books[isbn].sold_price += price * copies

        day, month, year = date.split("/")
        month = int(month)
        year = int(year)

        if (year, month) not in month_sold_books:
            month_sold_books[(year, month)] = 0
        month_sold_books[(year, month)] += copies

    file.close()

    print ("Available copies:")
    for isbn in sorted(books):
        if books[isbn].bought_copies - books[isbn].sold_copies > 0:
            print("\t%s: %d" % (isbn, books[isbn].bought_copies - books[isbn].sold_copies))

    print ("Sold books per month:")
    for year, month in sorted(month_sold_books):
        print("\t%s, %s: %d" % (month_to_string(month), str(year), month_sold_books[(year, month)]))

    print ("Gain per book:")
    for isbn in sorted(books):
        if books[isbn].sold_copies > 0:
            sold_price = books[isbn].sold_price
            sold_copies = books[isbn].sold_copies
            avg_bought_price = books[isbn].bought_price/books[isbn].bought_copies

            print("\t%s: %.1f (avg %.1f, sold %d)" % (
                isbn,
                sold_price - avg_bought_price * sold_copies,
                sold_price / sold_copies - avg_bought_price,
                sold_copies))