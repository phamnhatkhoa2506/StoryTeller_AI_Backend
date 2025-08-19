import unittest

from src.chains import *


class TestChain(unittest.TestCase):

    def test_other_chain(self):
        chain = OtherChain()

        response = chain.answer("Hello")

        print(response)


if __name__ == "__main__":
    unittest.main()