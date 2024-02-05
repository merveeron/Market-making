# Market-making
Market Making in Cryptocurrencies (Adapted Avellaneda &amp; Stoikov Strategy)

#MARKET VOLATILITY

def plot_data_with_events_in_range(dataframe, events_dataframe, start_date, end_date, display_dates): start_date = pd.to_datetime(start_date) end_date = pd.to_datetime(end_date) display_dates = pd.to_datetime(display_dates)

filtered_df = dataframe[(dataframe.index >= start_date) & (dataframe.index <= end_date)]
filtered_events = events_dataframe[(events_dataframe.index >= start_date) & (events_dataframe.index <= end_date)]

fig, ax1 = plt.subplots(figsize=(15, 8))  

ax1.set_xlabel('Date', fontsize=14, fontweight='bold')
ax1.set_ylabel('Close Price', color='tab:blue', fontsize=14, fontweight='bold')
ax1.plot(filtered_df.index, filtered_df['close'], color='tab:blue')
ax1.tick_params(axis='y', labelcolor='tab:blue', labelsize=12)

ax2 = ax1.twinx()
ax2.set_ylabel('Volatility', color='tab:red', fontsize=14, fontweight='bold')
ax2.plot(filtered_df.index, filtered_df['volatility'], color='tab:red', alpha=0.5)
ax2.tick_params(axis='y', labelcolor='tab:red', labelsize=12)

for idx, row in filtered_events.iterrows():
    ax1.axvline(x=idx, color='green', linestyle='--', alpha=0.7)

ax1.set_xticks(display_dates)
ax1.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
plt.xticks(rotation=45, fontsize=12)  # Adjusted rotation and font size

ax1.legend(['Close Price'], loc='upper left', fontsize=12)
ax2.legend(['Volatility'], loc='upper right', fontsize=12)
plt.title('Market Data with Significant Events Highlighted', fontsize=16, fontweight='bold')

plt.tight_layout()
plt.show()
display_dates = ['2024-01-01', '2024-01-09', '2024-01-12'] plot_data_with_events_in_range(df_with_volatility, events, '2023-12-25', '2024-01-17', display_dates)

#BENCHMARK BOT

class TradingBot: def init(self, symbol, spread, order_quantity, alpha, T, initial_inventory, frozen_inventory): self.symbol = symbol self.spread = spread self.order_quantity = order_quantity self.inventory = initial_inventory
self.frozen_inventory = frozen_inventory self.active_inventory = initial_inventory - frozen_inventory
self.exchange = ccxt.binance({ 'apiKey': '', 'secret': '', 'enableRateLimit': True, 'options': { 'defaultType': 'spot' } }) self.exchange.set_sandbox_mode(True) self.total_cost = 0 self.total_revenue = 0 self.alpha = alpha
self.T = T # Terminal time self.market_volatility = 0.1
self.total_orders_placed = 0 self.total_orders_executed = 0 self.total_orders = 0 self.filled_orders = 0 self.partially_filled_orders = 0 self.open_orders = 0 self.start_time = time.time() self.pnl_records = [] self.inventory_records = [] self.mid_prices = [] self.bid_prices = [] self.ask_prices = []

def print_order_details(self, order, order_type):
    print(f"{order_type}:")
    print(f"Transaction Time: {order['datetime']}")
    print(f"Executed Qty: {order['filled']}")
    print(f"Cumulative Quote Qty: {order.get('cummulativeQuoteQty', 'N/A')}")
    print(f"Status: {order['status']}")
    print(f"Order Status: {order['info'].get('status', 'N/A')}")
    print(f"Remaining Quantity: {order.get('remaining', 'N/A')}")
    print(f"Price: {order['price']}")
    print(f"Amount: {order['amount']}")
    print(f"Cost: {order['cost']}")
    

def fetch_mid_price(self):
    ticker = self.exchange.fetch_ticker(self.symbol)
    return (ticker['bid'] + ticker['ask']) / 2

def exponential_utility(self, pnl):
    return -np.exp(-self.alpha * pnl)

def expected_utility_after_trade(self, potential_trade_pnl):
    new_pnl = self.calculate_pnl() + potential_trade_pnl
    return self.exponential_utility(new_pnl)

def trade_decision(self, mid_price):
    buy_price = mid_price - self.spread / 2
    sell_price = mid_price + self.spread / 2
    
    if buy_price >= sell_price:
        buy_price = mid_price - self.spread
        sell_price = mid_price + self.spread
        
    actions = []
    if self.inventory > 0:
        actions.append(('sell', sell_price))

    balance = self.exchange.fetch_balance()
    free_balance = balance['free']['USDT']  
    if buy_price * self.order_quantity <= free_balance:
        actions.append(('buy', buy_price))
        
    self.mid_prices.append(mid_price)  
    self.bid_prices.append(buy_price)  
    self.ask_prices.append(sell_price)  

    potential_buy_pnl = -buy_price * self.order_quantity  
    potential_sell_pnl = sell_price * self.order_quantity  

    buy_utility = self.expected_utility_after_trade(potential_buy_pnl)
    sell_utility = self.expected_utility_after_trade(potential_sell_pnl)

    if buy_utility > sell_utility:
        actions.append(('buy', buy_price))
    else:
        actions.append(('sell', sell_price))

    return actions

def check_order_status(self, order_id):
    try:
        order = self.exchange.fetch_order(order_id, self.symbol)
        if order['status'] == 'closed':
            return True, order
        return False, order
    except Exception as e:
        print(f"Error checking order status: {e}")
        return False, None

def check_order_execution(self, order_id, side):
    order = self.exchange.fetch_order(order_id, self.symbol)
    if order['status'] == 'closed':
        return True, order
    return False, None

def place_order(self, side, price):
    if side == 'buy':
        balance = self.exchange.fetch_balance()
        free_balance = balance['free']['USDT']  # Change as per your base currency
        if price * self.order_quantity > free_balance:
            print("Not enough balance to buy. Skipping this order.")
            return

    if side == 'sell' and self.active_inventory < self.order_quantity:
        print("Not enough active inventory to sell. Skipping this order.")
        return


    try:
        order = self.exchange.create_limit_order(self.symbol, side, self.order_quantity, price)
        self.total_orders_placed += 1  # Increment total orders placed
        self.total_orders += 1 
      
        executed, order_info = self.check_order_execution(order['id'], side)
        if executed:
            self.total_orders_executed += 1  
            fee_cost = self.update_inventory_and_pnl(order_info, side)
            print(f"Transaction fee for {side} order: {fee_cost}")
            self.print_order_details(order_info, side.capitalize())  
            if order_info['filled'] == order_info['amount']:
                self.filled_orders += 1  
            else:
                self.partially_filled_orders += 1
        else:
            self.open_orders += 1  
            print(f"Order {side.capitalize()} not executed yet.")
    except Exception as e:
        print(f"Error placing order: {e}")
        
def update_inventory_and_pnl(self, order_info, side):
    executed_qty = float(order_info['filled'])
    executed_price = float(order_info['price'])
    fee_rate = 0.00075  
    fee_cost = executed_qty * executed_price * fee_rate

    prev_total_cost = self.total_cost
    prev_total_revenue = self.total_revenue
    prev_inventory = self.inventory

    if side == 'buy':
        self.inventory += executed_qty
        self.active_inventory += executed_qty
        self.total_cost += executed_qty * executed_price + fee_cost
    elif side == 'sell':
        self.inventory -= executed_qty
        self.active_inventory -= executed_qty
        self.total_revenue += executed_qty * executed_price - fee_cost

    trade_pnl = (executed_qty * executed_price - fee_cost) * (-1 if side == 'buy' else 1)
    self.pnl_records.append(trade_pnl)

    self.inventory_records.append(self.inventory)
    return fee_cost

    current_pnl = self.calculate_pnl()
    self.pnl_records.append(current_pnl)
    self.inventory_records.append(self.inventory)
    return fee_cost

def calculate_pnl(self):
    return self.total_revenue - self.total_cost

def run(self):
    while True:
        mid_price = self.fetch_mid_price()
        actions = self.trade_decision(mid_price)
        for action, price in actions:
            self.place_order(action, price)

        print(f"Current P&L: {self.calculate_pnl()}")
        print(f"Current Inventory: {self.inventory}")

        elapsed_time = time.time() - self.start_time
        if elapsed_time > self.T:
            execution_rate = self.total_orders_executed / self.total_orders_placed if self.total_orders_placed > 0 else 0
            print(f"Execution Rate: {execution_rate:.2f}")
            # Calculate and display the accumulated total P&L and inventory
            final_inventory = self.inventory_records[-1] if self.inventory_records else self.inventory
            accumulated_total_pnl = sum(self.pnl_records)
            
            print(f"Accumulated Total P&L: {accumulated_total_pnl}")
            print(f"Final Inventory: {final_inventory}")
            # Print order statistics
            print(f"Total Orders: {self.total_orders}")
            print(f"Filled Orders: {self.filled_orders}")
            print(f"Partially Filled Orders: {self.partially_filled_orders}")
            print(f"Open Orders: {self.open_orders}")
            
            avg_pnl = np.mean(self.pnl_records) if self.pnl_records else 0
            std_pnl = np.std(self.pnl_records) if self.pnl_records else 0
            avg_inventory_change = np.mean(self.inventory_records) if self.inventory_records else 0
            std_inventory_change = np.std(self.inventory_records) if self.inventory_records else 0

            print(f"Average P&L per Trade: {avg_pnl}")
            print(f"Standard Deviation of P&L: {std_pnl}")
            print(f"Average Inventory Change per Trade: {avg_inventory_change}")
            print(f"Standard Deviation of Inventory Change: {std_inventory_change}")
            break
            
    return self.mid_prices, self.bid_prices, self.ask_prices, self.pnl_records, self.inventory_records
            
    time.sleep(60)
frozen_inventory_amount = 0.01
bot = TradingBot( 'BTC/USDT', spread=0.005, order_quantity=0.001, alpha=0.5, T=60*30, initial_inventory=0.05, frozen_inventory=frozen_inventory_amount ) mid_prices, bid_prices, ask_prices, pnl_records, inventory_records = bot.run()

#MARKET IMPACT, DECAY RATE

limit = 1000000

since = binance_testnet.parse8601('2024-01-16T00:00:00Z') order_book = binance_testnet.fetch_order_book(symbol, limit) trades = binance_testnet.fetch_trades(symbol, since=since, limit=limit)

def calculate_mid_price(order_book): best_bid = order_book['bids'][0][0] if len(order_book['bids']) > 0 else None best_ask = order_book['asks'][0][0] if len(order_book['asks']) > 0 else None mid_price = (best_bid + best_ask) / 2 if best_bid and best_ask else None return mid_price

def analyze_execution_probability(order_book, depth=5): mid_price = calculate_mid_price(order_book) bid_distances = [abs(mid_price - bid[0]) for bid in order_book['bids'][:depth]] ask_distances = [abs(ask[0] - mid_price) for ask in order_book['asks'][:depth]] avg_bid_distance = np.mean(bid_distances) avg_ask_distance = np.mean(ask_distances) return avg_bid_distance, avg_ask_distance

def analyze_market_impact(trades, percentile=95, time_lag=5): trade_amounts = [trade['amount'] for trade in trades] large_trade_threshold = np.percentile(trade_amounts, percentile) impacts = [] for i in range(len(trades) - time_lag): if trades[i]['amount'] > large_trade_threshold: price_movement = trades[i + time_lag]['price'] - trades[i]['price'] impacts.append(price_movement) average_impact = np.mean(impacts) if impacts else 0 return average_impact

mid_price = calculate_mid_price(order_book) avg_bid_distance, avg_ask_distance = analyze_execution_probability(order_book) print(f"Average Bid Distance: {avg_bid_distance}, Average Ask Distance: {avg_ask_distance}")

decay_rate = 0.1 / (avg_bid_distance + avg_ask_distance) print(f"Decay Rate: {decay_rate}")

average_impact = analyze_market_impact(trades, percentile=95, time_lag=5) impact_factor = average_impact if average_impact != 0 else some_default_value print(f"Average Market Impact: {average_impact}") print(f"Impact Factor: {impact_factor}")

model_params = {'decay_rate': decay_rate} market_params = {'impact_factor': impact_factor}

#MEDIAN MARKET IMPACT AND DECAY RATE TO COMPARE AND MITIGATE THE RISK

def fetch_recent_trades(symbol, limit=1000000): return binance_testnet.fetch_trades(symbol, limit=limit)

def calculate_mid_price(order_book): best_bid = order_book['bids'][0][0] if len(order_book['bids']) > 0 else None best_ask = order_book['asks'][0][0] if len(order_book['asks']) > 0 else None return (best_bid + best_ask) / 2 if best_bid and best_ask else None

def analyze_execution_probability(order_book, depth=5): mid_price = calculate_mid_price(order_book) bid_distances = [abs(mid_price - bid[0]) for bid in order_book['bids'][:depth]] ask_distances = [abs(ask[0] - mid_price) for ask in order_book['asks'][:depth]] median_bid_distance = np.median(bid_distances) median_ask_distance = np.median(ask_distances) return median_bid_distance, median_ask_distance

def analyze_market_impact(trades, percentile=95, time_lag=5): trade_amounts = [trade['amount'] for trade in trades] large_trade_threshold = np.percentile(trade_amounts, percentile) impacts = [trades[i + time_lag]['price'] - trades[i]['price'] for i in range(len(trades) - time_lag) if trades[i]['amount'] >= large_trade_threshold] median_impact = np.median(impacts) if impacts else 0 return median_impact

order_book = binance_testnet.fetch_order_book(symbol) recent_trades = fetch_recent_trades(symbol)

mid_price = calculate_mid_price(order_book) median_bid_distance, median_ask_distance = analyze_execution_probability(order_book) decay_rate = 0.1 / (median_bid_distance + median_ask_distance) median_impact = analyze_market_impact(recent_trades)

print(f"Mid Price: {mid_price}") print(f"Median Bid Distance: {median_bid_distance}, Median Ask Distance: {median_ask_distance}") print(f"Decay Rate: {decay_rate}") print(f"Median Market Impact: {median_impact}")

model_params = {'decay_rate': decay_rate} market_params = {'impact_factor': median_impact}

#STRATEGY BOT

import ccxt import numpy as np import time from time import sleep

symbol = 'BTC/USDT' transaction_fee_rate = 0.00075

binance_testnet = ccxt.binance({ 'apiKey': '', 'secret': '', 'enableRateLimit': True, 'options': { 'defaultType': 'spot' } })

binance_testnet.set_sandbox_mode(True)

def robust_api_call(call, max_attempts=5, backoff_factor=1.5): attempt = 0 while attempt < max_attempts: try: result = call() if result is not None: return result except ccxt.NetworkError as e: print(f"Network error: {e}. Retrying...") sleep(backoff_factor ** attempt) except ccxt.BaseError as e: print(f"CCXT Base error: {e}. Retrying...") sleep(backoff_factor ** attempt) except Exception as e: print(f"An unexpected error occurred: {e}") break finally: attempt += 1 return None

def getminutedata(symbol, interval, lookback): ohlcv = binance_testnet.fetch_ohlcv(symbol, timeframe=interval, since=binance_testnet.parse8601(lookback)) frame = pd.DataFrame(ohlcv, columns=['Time', 'Open', 'High', 'Low', 'Close', 'Volume']) frame = frame.set_index('Time') frame.index = pd.to_datetime(frame.index, unit='ms') frame = frame.astype(float) return frame

def calculate_volatility(symbol, interval='1m', lookback='30m'): historical_data = getminutedata(symbol, interval, lookback) historical_data['log_returns'] = np.log(historical_data['Close'] / historical_data['Close'].shift(1)) volatility = historical_data['log_returns'].std() print(f"Calculated Volatility: {volatility}") return volatility

def exponential_utility(pnl, alpha): return -np.exp(-alpha * pnl)

def calculate_reservation_price(s, q, gamma, sigma, T_minus_t): return s - q * gamma * sigma**2 * T_minus_t

def calculate_optimal_spread(gamma, sigma, T_minus_t, kappa): return gamma * sigma**2 * T_minus_t + (2 / gamma) * np.log(1 + gamma / kappa)

def get_current_market_price(symbol): ticker = binance_testnet.fetch_ticker(symbol) mid_price = (ticker['bid'] + ticker['ask']) / 2 return mid_price

def place_limit_orders(bid_price, ask_price, adapted_order_size, current_inventory, mid_price, model_params, market_params, symbol, alpha): bid_distance = mid_price - bid_price ask_distance = ask_price - mid_price bid_execution_prob = calculate_execution_probability(bid_distance, model_params) ask_execution_prob = calculate_execution_probability(ask_distance, model_params) bid_market_impact = calculate_market_impact(adapted_order_size, market_params) ask_market_impact = calculate_market_impact(adapted_order_size, market_params)

adjusted_buy_amount = max(0, adapted_order_size * bid_execution_prob - bid_market_impact)
adjusted_sell_amount = max(0, adapted_order_size * ask_execution_prob - ask_market_impact)

print(f"Adjusted Buy Amount: {adjusted_buy_amount}, Adjusted Sell Amount: {adjusted_sell_amount}")

if ask_price < bid_price:
    print("Sell price is lower than buy price. Not executing sell order.")
    adjusted_sell_amount = 0

potential_buy_pnl = -bid_price * adjusted_buy_amount
potential_sell_pnl = ask_price * adjusted_sell_amount
buy_utility = exponential_utility(potential_buy_pnl, alpha)
sell_utility = exponential_utility(potential_sell_pnl, alpha)

buy_order_id, sell_order_id = None, None
if buy_utility > sell_utility and adjusted_buy_amount > 0:
    try:
        buy_order = binance_testnet.create_limit_buy_order(symbol, adjusted_buy_amount, bid_price)
        print_order_details(buy_order, "Buy Order")
        buy_order_id = buy_order['id']
    except Exception as e:
        print(f"An error occurred while placing buy order: {e}")
elif sell_utility > buy_utility and adjusted_sell_amount > 0 and current_inventory >= adjusted_sell_amount:
    try:
        sell_order = binance_testnet.create_limit_sell_order(symbol, adjusted_sell_amount, ask_price)
        print_order_details(sell_order, "Sell Order")
        sell_order_id = sell_order['id']
    except Exception as e:
        print(f"An error occurred while placing sell order: {e}")

return buy_order_id, sell_order_id
    
print_order_details(buy_order, "Buy Order")

print_order_details(sell_order, "Sell Order")
def print_order_details(order, order_type): print(f"{order_type}:") print(f"Transaction Time: {order['datetime']}") print(f"Executed Qty: {order['filled']}") print(f"Status: {order['status']}") print(f"Order Status: {order['info'].get('status', 'N/A')}") print(f"Remaining Quantity: {order.get('remaining', 'N/A')}") print(f"Price: {order['price']}") print(f"Amount: {order['amount']}") print(f"Cost: {order['cost']}")

executed_qty = float(order['filled'])
executed_price = float(order['price'])
transaction_amount = executed_qty * executed_price
transaction_fee = transaction_amount * transaction_fee_rate
print(f"Calculated Transaction Fee: {transaction_fee}")
def cancel_all_orders(symbol): orders = binance_testnet.fetch_open_orders(symbol) for order in orders: binance_testnet.cancel_order(order['id'], symbol)

def check_orders_and_calculate_pnl(order_ids): pnl = 0 transaction_costs = 0 for order_id in order_ids: order = binance_testnet.fetch_order(order_id, symbol) if order['status'] in ['closed', 'partially_filled']: executed_qty = float(order['filled']) executed_price = float(order['price']) transaction_amount = executed_qty * executed_price cost = transaction_amount * transaction_fee_rate

        if order['side'] == 'buy':
            pnl -= (transaction_amount + cost)
        else:
            pnl += (transaction_amount - cost)

        transaction_costs += cost

return pnl, transaction_costs
def update_pnl_and_inventory(order_id, current_inventory, current_pnl, symbol): order = robust_api_call(lambda: binance_testnet.fetch_order(order_id, symbol)) if order and order['status'] in ['closed', 'partially_filled']: executed_qty = float(order['filled']) executed_price = float(order['price']) transaction_amount = executed_qty * executed_price transaction_fee = transaction_amount * transaction_fee_rate

    if order['side'] == 'buy':
        current_inventory += executed_qty
        current_pnl -= (transaction_amount + transaction_fee)
    else: 
        current_inventory -= executed_qty
        current_pnl += (transaction_amount - transaction_fee)

return current_inventory, current_pnl
def get_account_balance(): balance = binance_testnet.fetch_balance() usdt_balance = balance['total']['USDT']
return usdt_balance

def dynamic_order_size(current_inventory, max_inventory, volatility, account_balance, mid_price, base_order_size=0.005, max_order_size=0.01, frozen_inventory_ratio=0.2):

frozen_inventory = max_inventory * frozen_inventory_ratio
available_inventory = max(current_inventory - frozen_inventory, 0)
inventory_factor = max(1 - abs(available_inventory) / (max_inventory - frozen_inventory), 0)
volatility_factor = min(volatility * 10, 1)  

dynamic_size = base_order_size + (max_order_size - base_order_size) * volatility_factor * inventory_factor

max_order_value = account_balance * 0.1 
max_order_size_by_value = max_order_value / mid_price 

return min(dynamic_size, max_order_size, max_order_size_by_value)
def adjust_gamma(base_gamma, time_elapsed, total_time): remaining_time_fraction = (total_time - time_elapsed) / total_time return base_gamma * remaining_time_fraction

def calculate_execution_probability(distance, model_params): decay_rate = model_params.get('decay_rate', 0.04690431519695075) probability = np.exp(-decay_rate * abs(distance)) return probability

model_params = {'decay_rate': 0.007655795437145341} market_params = {'impact_factor': 0.657692307692345}

def run_strategy(initial_inventory=0.05): T = 60 current_inventory = initial_inventory current_pnl = 0 total_pnl = 0 base_gamma = 0.1 kappa = 1.5 max_inventory = 10 T_minus_t = T

alpha = 
utility_record = []

mid_prices, bid_prices, ask_prices, inventory_levels, pnl_records = [], [], [], [], []

start_time = time.time()

total_orders_placed = 0
total_orders_executed = 0
total_orders_initialized = 0
total_orders_filled = 0
total_orders_partially_filled = 0
total_orders_open = 0

while time.time() - start_time < T:
    try:
        mid_price = robust_api_call(lambda: get_current_market_price(symbol))
        print(f"Mid-Price: {mid_price}")
        if mid_price is None:
            print("Failed to fetch Mid-Price")
            continue

        sigma = robust_api_call(lambda: calculate_volatility(symbol, '1m', '30m'))
        if sigma is None:
            print("Failed to calculate volatility")
            continue  
        account_balance = robust_api_call(get_account_balance)
        if account_balance is None:
            print("Failed to fetch account balance")
            continue
        else:
            print("Debug - Account Balance:", account_balance)

        time_elapsed = time.time() - start_time
        gamma = adjust_gamma(base_gamma, time_elapsed, T)
        print(f"Adjusted Gamma: {gamma}")

        reservation_price = robust_api_call(lambda: calculate_reservation_price(mid_price, current_inventory, gamma, sigma, T_minus_t))
        spread = robust_api_call(lambda: calculate_optimal_spread(gamma, sigma, T, kappa))
        bid_price = reservation_price - spread / 2
        ask_price = reservation_price + spread / 2
        

        adapted_order_size = dynamic_order_size(current_inventory, max_inventory, sigma, account_balance, mid_price)
        print(f"Calculated Adapted Order Size: {adapted_order_size}")
        if adapted_order_size is None:
            print("Failed to calculate adapted order size")
            continue
            
        buy_order_id, sell_order_id = place_limit_orders(bid_price, ask_price, adapted_order_size, current_inventory, mid_price, model_params, market_params, symbol, alpha)

        temp_pnl = current_pnl  
        temp_inventory = current_inventory  
        for order_id in [buy_order_id, sell_order_id]:
            if order_id is not None:
                total_orders_initialized += 1
                temp_inventory, temp_pnl = update_pnl_and_inventory(order_id, temp_inventory, temp_pnl, symbol)
                order = robust_api_call(lambda: binance_testnet.fetch_order(order_id, symbol))
                if order and order['status'] == 'closed':
                    total_orders_filled += 1
                elif order and order['status'] == 'partially_filled':
                    total_orders_partially_filled += 1
                elif order and order['status'] == 'open':
                    total_orders_open += 1
                
        current_pnl = temp_pnl
        current_inventory = temp_inventory
        print(f"Current P&L: {current_pnl}, Current Inventory: {current_inventory}")
        pnl_records.append(current_pnl)
        inventory_levels.append(current_inventory)

        current_utility = exponential_utility(current_pnl, alpha)
        utility_record.append(current_utility)
        print(f"Current Utility: {current_utility}")

        mid_prices.append(mid_price)
        bid_prices.append(bid_price)
        ask_prices.append(ask_price)

        T_minus_t -= 1
        time.sleep(1)
        
    except Exception as e:
        print(f"An error occurred: {e}")
        time.sleep(5)

total_pnl = current_pnl  

execution_rate = total_orders_executed / total_orders_placed if total_orders_placed > 0 else 0
avg_pnl = np.mean(pnl_records) if pnl_records else 0
std_pnl = np.std(pnl_records) if pnl_records else 0
avg_inventory = np.mean(inventory_levels) if inventory_levels else 0
std_inventory = np.std(inventory_levels) if inventory_levels else 0

print(f"Final Total P&L: {total_pnl}, Final Inventory: {current_inventory}")
print(f"Average P&L: {avg_pnl}, Standard Deviation of P&L: {std_pnl}")
print(f"Average Inventory: {avg_inventory}, Standard Deviation of Inventory: {std_inventory}")
print(f"Total Orders Initialized: {total_orders_initialized}")
print(f"Total Orders Filled: {total_orders_filled}")
print(f"Total Orders Partially Filled: {total_orders_partially_filled}")
print(f"Total Orders Open: {total_orders_open}")

return mid_prices, bid_prices, ask_prices, inventory_levels, pnl_records, execution_rate
mid_prices, bid_prices, ask_prices, inventory_levels, pnl_records, execution_rate = run_strategy(initial_inventory=0.05)
