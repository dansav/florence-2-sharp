using Avalonia.Controls;
using Avalonia.Input;
using Avalonia.Interactivity;
using Avalonia.Platform.Storage;

namespace FlorenceTwoLab.Desktop;

public partial class MainWindow : Window
{
    public MainWindow()
    {
        InitializeComponent();
        
        DataContext = new MainViewModel();

        Loaded += MainWindow_Loaded;
        
        AddHandler(DragDrop.DragOverEvent, DragOver);
        AddHandler(DragDrop.DropEvent, Drop);
    }

    private async void Drop(object? sender, DragEventArgs e)
    {
        if (e.Data.Contains(DataFormats.Files) && DataContext is MainViewModel viewModel)
        {
            var items = e.Data.GetFiles() ?? System.Array.Empty<IStorageItem>();
            foreach (var item in items)
            {
                if (item is IStorageFile file)
                {
                        await using var stream = await file.OpenReadAsync();
                        await viewModel.LoadImageAsync(stream);
                        return;
                }
                
                if (item is IStorageFolder folder)
                {
                    await foreach (var item2 in folder.GetItemsAsync())
                    {
                        if (item2 is IStorageFile file2)
                        {
                            await using var stream = await file2.OpenReadAsync();
                            await viewModel.LoadImageAsync(stream);
                            return;
                        }
                    }
                }
            }
        }
    }

    private void DragOver(object? sender, DragEventArgs e)
    {
        
    }

    private void MainWindow_Loaded(object? sender, RoutedEventArgs e)
    {
        if (DataContext is MainViewModel viewModel)
        {
           // viewModel.All();
        }
    }
}

