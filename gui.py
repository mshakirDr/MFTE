import tkinter
import sys
from MFTE import tag_MD, tag_MD_parallel, tag_stanford, tag_stanford_stanza, do_counts
import os
import timeit
import threading

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

check_button_state = True
check_button_state2 = False
ttr_value = 400


def call_MFTE(folder_selected: str) -> None:
    """Call MFTE functions after directory selection

    Args:
        folder_selected (str): dir path returned by filedialog
    """
        
    global check_button_state, ttr_value, check_button_state2
    ###############################################
    # #input_dir = r"/Users/Elen/Documents/PhD/Publications/2023_Shakir_LeFoll/test_files/"
    # input_dir = folder_selected + "/"
    # # download Stanford CoreNLP and unzip in this directory. See this page #https://stanfordnlp.github.io/stanza/client_setup.html#manual-installation
    # # direct download page https://stanfordnlp.github.io/CoreNLP/download.html
    # nlp_dir = r"/Users/Elen/Documents/PhD/Publications/2023_Shakir_LeFoll/stanford-corenlp-4.5.1/"
    # output_stanford = os.path.dirname(input_dir.rstrip("/").rstrip("\\")) + "/" + os.path.basename(input_dir.rstrip("/").rstrip("\\")) + "_MFTE_StanfordPOS/"
    # output_MD = os.path.dirname(input_dir.rstrip("/").rstrip("\\")) + "/" + os.path.basename(input_dir.rstrip("/").rstrip("\\")) + "_MFTE_Tagged/"
    # output_stats = os.path.dirname(input_dir.rstrip("/").rstrip("\\")) + "/" + os.path.basename(input_dir.rstrip("/").rstrip("\\")) + "_MFTE_Counts/"
    # if isinstance(ttr_value, int):
    #     ttr = ttr_value
    # else:
    #     ttr = 400
    # # tag_stanford(nlp_dir, input_dir, output_stanford)
    # t_0 = timeit.default_timer()
    # tag_stanford_stanza(input_dir, output_stanford)
    # t_1 = timeit.default_timer()
    # elapsed_time = round((t_1 - t_0) * 10 ** 6, 3)
    # print("Time spent on tagging process (micro seconds):", elapsed_time)
    # tag_MD(output_stanford, output_MD, extended=check_button_state)
    # # tag_MD_parallel(output_stanford, output_MD, extended=check_button_state)
    # do_counts(output_MD, output_stats, ttr)

    input_dir = folder_selected + "/"
    output_stanford = os.path.dirname(input_dir.rstrip("/").rstrip("\\")) + "/" + os.path.basename(input_dir.rstrip("/").rstrip("\\")) + "_MFTE_tagged_test/"
    output_MD = output_stanford + "MD/"
    output_stats = output_MD + "Statistics/"
    if isinstance(ttr_value, int):
        ttr = ttr_value
    else:
        ttr = 400
    # record start time
    t_0 = timeit.default_timer()
    tag_stanford_stanza(input_dir, output_stanford)
    #tag_stanford(nlp_dir, input_dir, output_stanford)
    t_1 = timeit.default_timer()
    elapsed_time = round((t_1 - t_0) * 10 ** 6, 3)
    print("Time spent on tagging process (micro seconds):", elapsed_time)
    #parallel MD tag
    if check_button_state2:
        tag_MD_parallel(output_stanford, output_MD, extended=check_button_state)
    #otherwise simple MD
    else:
        tag_MD(output_stanford, output_MD, extended=check_button_state)
    #
    do_counts(output_MD, output_stats, ttr)

def button_function() -> None:
    """Select corpus directory and apply MFTE tagging on the files in it
    """
    from tkinter import filedialog
    button.configure(command=threading.Thread(target=button_function).start)
    folder_selected = filedialog.askdirectory(title="Select corpus files directory")
    if folder_selected != "":
        print("selected directory:", folder_selected)
        call_MFTE(folder_selected)
    else:
        print("Sorry no directory was selected...")
    

#update extended tag boolean
def checkbox_event():
    print("Tag extended:", check_var.get())
    check_button_state = check_var.get()

#update parallel MD tag boolean
def checkbox_event2():
    print("Parellel MD tag:", check_var2.get())
    check_button_state2 = check_var2.get()

#update ttr value
def entrybox_event(*args):
    ttr_value = v.get()
    print("ttr:", ttr_value)

if __name__ == "__main__":
    app = tkinter.Tk()  # create CTk window like you do with the Tk window
    app.geometry("600x350")
    app.title("MFTE")
    # this removes the maximize button
    app.resizable(0,0)
    button = tkinter.Button(master=app, text="Select corpus directory", command=threading.Thread(target=button_function).start)
    button.place(x=10, y=20)
    ttr_lable = tkinter.Label(app, text = "TTR").place(x = 400, y = 20)
    v = tkinter.StringVar(app, value='400')
    v.trace_add('write', entrybox_event)
    ttr_entry = tkinter.Entry(master=app, textvariable=v, validatecommand=entrybox_event).place(x=450, y=20)
    check_var = tkinter.BooleanVar(app, True)
    checkbox = tkinter.Checkbutton(master=app, text="Extended tags", command=checkbox_event,
                                        variable=check_var, onvalue=True, offvalue=False)
    checkbox.place(x=250, y=20)
    checkbox.select()
    check_var2 = tkinter.BooleanVar(app, False)
    checkbox2 = tkinter.Checkbutton(master=app, text="Parallel MD tagging", command=checkbox_event2,
                                        variable=check_var2, onvalue=True, offvalue=False)
    checkbox2.place(x=10, y=50)
    checkbox2.deselect()
    #get True or False for extended
    check_button_state = check_var.get()
    #get ttr value
    ttr_value = v.get()
    frame = tkinter.Frame(app, width=580, height=250)
    frame.pack()
    frame.place(x=10, y=90)
    console = Console(master=frame, height=15).place(x=0, y=0)
    app.mainloop()