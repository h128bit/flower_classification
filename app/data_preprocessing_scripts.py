from torchvision.transforms import v2

preprocessing = v2.Compose([v2.ToPILImage(),
                            v2.Resize(size=(64, 64)),
                            v2.ToTensor(),
                            v2.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
                            ])
