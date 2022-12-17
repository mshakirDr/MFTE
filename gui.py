import tkinter
import sys
#https://www.reddit.com/r/Tkinter/comments/nmx0ir/how_to_show_terminal_output_in_gui/
class Console(tkinter.Text):
    def __init__(self, *args, **kwargs):
        kwargs.update({"state": "disabled"})
        tkinter.Text.__init__(self, *args, **kwargs)
        self.bind("<Destroy>", self.reset)
        self.old_stdout = sys.stdout
        sys.stdout = self
    
    def delete(self, *args, **kwargs):
        self.config(state="normal")
        self.delete(*args, **kwargs)
        self.config(state="disabled")
    
    def write(self, content):
        self.config(state="normal")
        self.insert("end", content)
        self.config(state="disabled")
        self.see(tkinter.END)
        self.yview(tkinter.END)

    
    def reset(self, event):
        sys.stdout = self.old_stdout

def button_function():
    folder_selected = tkinter.filedialog.askdirectory(title="Select corpus files directory")
    print(folder_selected)


def checkbox_event():
    print("checkbox toggled, current value:", check_var.get())

app = tkinter.Tk()  # create CTk window like you do with the Tk window
app.geometry("600x250")
app.title("MFTE")
# this removes the maximize button
app.resizable(0,0)

# Use CTkButton instead of tkinter Button
button = tkinter.Button(master=app, text="Select corpus directory", command=button_function)
button.place(x=10, y=20)
ttr_lable = tkinter.Label(app, text = "TTR").place(x = 400, y = 20)
v = tkinter.StringVar(app, value='400')
ttr_entry = tkinter.Entry(master=app, textvariable=v).place(x=450, y=20)
check_var = tkinter.StringVar(app, "on")
checkbox = tkinter.Checkbutton(master=app, text="Extended tags", command=checkbox_event,
                                     variable=check_var, onvalue="on", offvalue="off").place(x=250, y=20)


frame = tkinter.Frame(app, width=580, height=170)
frame.pack()
frame.place(x=10, y=70)
console = Console(master=frame, height=10).place(x=0, y=0)


app.mainloop()