using Gpt4All;
using System.Diagnostics;

namespace GPT4AllChat
{
    public partial class GPT4AChat : Form
    {
        //variable declarations
        Size client = new();
        readonly TextBox tbPrompt = new();
        readonly Button bLoad = new();
        readonly RichTextBox rtbResults = new();
        bool modelLoaded = false;
        string modelPath = string.Empty;
        string prompt = string.Empty;
        IGpt4AllModel? model;
        Gpt4AllModelFactory modelFactory = new();
        PredictRequestOptions myopts = PredictRequestOptions.Defaults;
        nuint _logitssize = 0;
        nuint _tokenssize = 0;
        int _pastconversationtokensnum = 0;
        int _contextsize = 4096;
        int _tokenstopredict = 512;
        int _topk = 40;
        float _topp = 0.9f;
        float _temperature = 0.5f;
        int _batches = 128;
        float _repeatpenalty = 1.2f;
        int _repeatlastn = 128;
        float _contexterase = 0.0f;
        bool showprompt = true;
        bool showtime = true;
        Stopwatch swatch = new();
        public GPT4AChat()
        {
            InitializeComponent();
        }

        private async void DoLoadModel(object? sender, EventArgs e)
        {
            if (!modelLoaded) //if model is not loaded
            {
                bool gotfile = false; //we have not gotten file loaded
                int numtries = 0;
                while (!gotfile) //while we don't have file loaded
                {
                    if (tbPrompt.Text != string.Empty) //if there is something in the prompt area
                    {
                        modelPath = tbPrompt.Text; //we'll try using it as our model path
                        tbPrompt.Text = "";
                    }
                    else //otherwise
                    {
                        using OpenFileDialog openFileDialog = new(); //we'll need an openfile dialog
                        openFileDialog.Filter = "Model files (*.bin)|*.bin|All files (*.*)|*.*";
                        openFileDialog.FilterIndex = 2;
                        openFileDialog.RestoreDirectory = true;
                        if (openFileDialog.ShowDialog() == DialogResult.OK)
                            modelPath = openFileDialog.FileName;
                    }
                    if (numtries++ > 5) //if we've tried 5 times
                    {
                        //show error and quit
                        MessageBox.Show("Cannot load model!  Exiting...", "ERROR", MessageBoxButtons.OK, MessageBoxIcon.Error);
                        Close();
                    }
                    try
                    {
                        //modelFactory = new Gpt4AllModelFactory(); //get model factory
                        model = modelFactory.LoadModel(modelPath); //try loading the model
                        if (model != null) //if we got it
                        {
                            bLoad.Text = "Prompt"; //change button prompt
                            rtbResults.Clear();
                            ShowHelp(); //show help
                            gotfile = modelLoaded = true; //set flags
                        }
                    }
                    catch
                    {
                        //show error
                        MessageBox.Show("Error loading model: " + modelPath + "!", "ERROR!", MessageBoxButtons.OK, MessageBoxIcon.Error);
                    }
                }
            }
            else //otherwise (model is loaded)
            {
                prompt = tbPrompt.Text; //get the prompt
                if (prompt[0] == '/') //if prompt starts with a slash
                    SetIt(prompt); //deal with it
                else
                {
                    if (showprompt)
                        rtbResults.AppendText("\n> " + prompt + " >\n\n"); //display prompt in results area
                    if (model == null) return;
                    bLoad.Enabled = false; //disable button
                    bool allatonce = true;
                    if (showtime)
                        swatch.Start();
                    if (allatonce) //if we want all the response at once
                    {
                        var response = await model.GetPredictionAsync(prompt, myopts); //get response
                        if (response.Success)
                        {
                            try
                            {
                                string result;
                                result = await response.GetPredictionAsync(); //get response
                                rtbResults.AppendText(result + "\n"); //display it
                            }
                            catch
                            {
                                //display error
                                rtbResults.AppendText("Unexpected Error getting response!\n\n");
                            }
                        }
                        else
                        {
                            //display error
                            rtbResults.AppendText("Sorry, your prompt of \"" + prompt + "\" resulted in an error!\n" + response.ErrorMessage + "\n\n");
                        }
                    }
                    else //otherwise (streaming response)
                    {
                        var result = await model.GetStreamingPredictionAsync(prompt, myopts); //get response
                        if (result.Success)
                        {
                            await foreach (var token in result.GetPredictionStreamingAsync()) //display each token
                                rtbResults.AppendText(token);
                            rtbResults.AppendText("\n");
                        }
                        else
                        {
                            //display error
                            rtbResults.AppendText("Sorry, your prompt of \"" + prompt + "\" resulted in an error!\n" + result.ErrorMessage + "\n\n");
                        }
                    }
                    if (showtime)
                    {
                        swatch.Stop();
                        rtbResults.AppendText("(" + swatch.ElapsedMilliseconds.ToString() + "ms)\n");
                    }
                    rtbResults.Invalidate();
                    bLoad.Enabled = true;
                }
            }
            tbPrompt.Text = "";
        }

        private void DoLoadGPT4AllChat(object sender, EventArgs e)
        {
            //set up controls
            Size = new(Screen.PrimaryScreen.Bounds.Width, Screen.PrimaryScreen.Bounds.Height); //set to desktop size
            Location = new Point(0, 0); //put in upper left corner of screen
            client = ClientSize;
            int margins = 10;
            tbPrompt.Location = new Point(margins, margins);
            tbPrompt.Size = new Size(client.Width - margins * 3 - 150, 35);
            bLoad.Location = new Point(client.Width - margins - 150, margins);
            bLoad.Size = new Size(150, 35);
            bLoad.Text = "Load Model";
            bLoad.Click += DoLoadModel;
            rtbResults.Location = new Point(margins, margins * 2 + 35);
            rtbResults.Size = new Size(client.Width - margins * 2, client.Height - margins * 3 - 35);
            Controls.Add(tbPrompt);
            Controls.Add(bLoad);
            Controls.Add(rtbResults);
            AcceptButton = bLoad;
            MakeOptions();
        }
        public void ShowHelp()
        {
            //display help text
            rtbResults.AppendText("\nGPT4All Chat\n\nCopyright 2023 by John Worthington\n\n");
            if (model != null)
                rtbResults.AppendText(model.ToString() + "\n\n");
            rtbResults.AppendText("Type /help for help.\n");
            rtbResults.AppendText("     /newmodel for a new model.\n");
            rtbResults.AppendText("     /set log # to set Logits Size to #.\n");
            rtbResults.AppendText("     /set toksiz # to set Tokens Size to #.\n");
            rtbResults.AppendText("     /set past # to set Past Conversation Tokens to #.\n");
            rtbResults.AppendText("     /set context # to set Context Size to #.\n");
            rtbResults.AppendText("     /set tokpre # to set Tokens To Predict to #.\n");
            rtbResults.AppendText("     /set topk # to set Top_K to #.\n");
            rtbResults.AppendText("     /set topp # to set Top_P to #.\n");
            rtbResults.AppendText("     /set temp # to set Temperature to #.\n");
            rtbResults.AppendText("     /set bat # to set Batches to #.\n");
            rtbResults.AppendText("     /set reppen # to set Repeat Penalty to #.\n");
            rtbResults.AppendText("     /set replast # to set Repeat Last N to #.\n");
            rtbResults.AppendText("     /set erase # to set Context Erase to #.\n");
            rtbResults.AppendText("     /values to get values of various parameters.\n");
            rtbResults.AppendText("     /clear to clear this area.\n");
            rtbResults.AppendText("     /exit to exit the program.\n");
        }

        static List<string> ParseIt(string command)
        {
            List<string> result = new(); //new string list
            result.Clear();
            while (command.IndexOf(' ') != -1) //while there is a space
            {
                result.Add(command[..command.IndexOf(' ')]); //add up to space to list
                command = command[(command.IndexOf(' ') + 1)..]; //cut off first part of command
            }
            result.Add(command); //add rest of command
            return result;
        }
        public void SetIt(string command)
        {
            List<string> comms = ParseIt(command); //parse command
            _logitssize = myopts.LogitsSize; //get variables
            _tokenssize = myopts.TokensSize;
            _pastconversationtokensnum = myopts.PastConversationTokensNum;
            _contextsize = myopts.ContextSize;
            _tokenstopredict = myopts.TokensToPredict;
            _topk = myopts.TopK;
            _topp = myopts.TopP;
            _temperature = myopts.Temperature;
            _batches = myopts.Batches;
            _repeatpenalty = myopts.RepeatPenalty;
            _repeatlastn = myopts.RepeatLastN;
            _contexterase = myopts.ContextErase;
            if (comms.Count == 3) //if command was in 3 parts, i.e. /set var #
            {
                try
                {
                    //set variables according to command
                    if (comms[1] == "log") _ = nuint.TryParse(comms[2], out _logitssize);
                    if (comms[1] == "toksiz") _ = nuint.TryParse(comms[2], out _tokenssize);
                    if (comms[1] == "past") _ = int.TryParse(comms[2], out _pastconversationtokensnum);
                    if (comms[1] == "context") _ = int.TryParse(comms[2], out _contextsize);
                    if (comms[1] == "tokpre") _ = int.TryParse(comms[2], out _tokenstopredict);
                    if (comms[1] == "topk") _ = int.TryParse(comms[2], out _topk);
                    if (comms[1] == "topp") _ = float.TryParse(comms[2], out _topp);
                    if (comms[1] == "temp") _ = float.TryParse(comms[2], out _temperature);
                    if (comms[1] == "bat") _ = int.TryParse(comms[2], out _batches);
                    if (comms[1] == "reppen") _ = float.TryParse(comms[2], out _repeatpenalty);
                    if (comms[1] == "replast") _ = int.TryParse(comms[2], out _repeatlastn);
                    if (comms[1] == "erase") _ = float.TryParse(comms[2], out _contexterase);
                }
                catch
                {
                    //if error
                    ShowHelp();
                    rtbResults.AppendText("\nError!  \"" + command + "\" (\"" + comms[0] + " " + comms[1] + " " + comms[2] + "\") is not valid!\n\n");
                    return; //we do not want to make options with possibly messed up data, so return
                }
                MakeOptions(); //make new options
            }
            else //otherwise
            {
                if (comms[0] == "/values") //if we want values of options
                {
                    //display them
                    rtbResults.AppendText("\nLogits Size: " + _logitssize.ToString() + "\n");
                    rtbResults.AppendText("Tokens Size: " + _tokenssize.ToString() + "\n");
                    rtbResults.AppendText("Past Conversation Tokens Number: " + _pastconversationtokensnum.ToString() + "\n");
                    rtbResults.AppendText("Context Size: " + _contextsize.ToString() + "\n");
                    rtbResults.AppendText("Tokens To Predict: " + _tokenstopredict.ToString() + "\n");
                    rtbResults.AppendText("Top K: " + _topk.ToString() + "\n");
                    rtbResults.AppendText("Top P: " + _topp.ToString() + "\n");
                    rtbResults.AppendText("Temperature: " + _temperature.ToString() + "\n");
                    rtbResults.AppendText("Batches: " + _batches.ToString() + "\n");
                    rtbResults.AppendText("Repeat Penalty: " + _repeatpenalty.ToString() + "\n");
                    rtbResults.AppendText("Repeat Last N: " + _repeatlastn.ToString() + "\n");
                    rtbResults.AppendText("Context Erase: " + _contexterase.ToString() + "\n\n");
                }
                if (comms[0] == "/clear") rtbResults.Clear(); //other commands
                if (comms[0] == "/exit") Close();
                if (comms[0] == "/help") ShowHelp();
                if (comms[0] == "/newmodel") //if we asked for a new model
                {
                    model?.Dispose();
                    bLoad.Text = "Load Model";
                    modelLoaded = false;
                    tbPrompt.Text = "";
                }
            }
        }

        private void MakeOptions()
        {
            myopts = new PredictRequestOptions() with
            {
                LogitsSize = _logitssize,
                TokensSize = _tokenssize,
                PastConversationTokensNum = _pastconversationtokensnum,
                ContextSize = _contextsize,
                TokensToPredict = _tokenstopredict,
                TopK = _topk,
                TopP = _topp,
                Temperature = _temperature,
                Batches = _batches,
                RepeatPenalty = _repeatpenalty,
                RepeatLastN = _repeatlastn,
                ContextErase = _contexterase
            };
        }
    }
}
