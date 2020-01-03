import IPython

class Grandparent(object):
    def __init__(self):
        self.id = 5

    def foo(self, num):
        return num + 1

    def my_method(self):
        print "Grandparent"
        print self.foo(self.id)

class Parent(Grandparent):
    def __init__(self):
        Grandparent.__init__(self)
        self.id = 100
    def parentCall(self):
        self.my_method()

    def some_other_method(self):
        print "Parent"

class Child(Grandparent):
    def __init__(self):
        self.id = 55555555
    def small_method(self):
        print "Hello Grandparent"
        print '---------'
        super(Child, self).my_method()
        print '----'
        print self.my_method()
        super(Parent, self).my_method()

if __name__ == '__main__':

    granny = Grandparent()
    mom = Parent()
    kid = Child()
    IPython.embed()
