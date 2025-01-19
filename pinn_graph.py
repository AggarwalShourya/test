xx=torch.linspace(-10,11, 1000)[:, None].float()
with torch.no_grad():
    yy = model(xx)

plt.figure(figsize=(10, 6))
plt.plot(xx.detach().numpy(), yy.detach().numpy(), label='Predicted', color='blue')
plt.title('Predicted Values vs Input x')
plt.xlabel('x')
plt.ylabel('Predicted y')
plt.legend()
plt.grid(True)
plt.show()
