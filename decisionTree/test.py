glb_var = "sdfsdf"

class TreeNode:
    def __init__(self,IG=0,left_node=None,right_node=None):
        self.IG = IG
        self.left_child = left_node
        self.right_child = right_node

    def __str__(self):
        return self.value

    def pprint_tree(self,depth=0):
        # todo:递归打印树
        if(depth==0):
            print("glb_var:%s"%glb_var)
        print("| "*(depth),end='')
        print("cur dep:%d"%depth)
        if self.left_child:
            self.left_child.pprint_tree(depth+1)
        if self.right_child:
            self.right_child.pprint_tree(depth+1)

if __name__ == '__main__':

    root = TreeNode()
    root.left_child = TreeNode()
    root.left_child.left_child = TreeNode()
    root.left_child.left_child.left_child = TreeNode()
    root.right_child = TreeNode()
    #glb_var = "1231232"
    root.pprint_tree()

    a = [['123'],['23'],['sgdf']]

    for line in a:
        for i in range(len(line)):
            line[i] = 3

    print(a)